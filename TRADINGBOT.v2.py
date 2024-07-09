import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque


class NDR_TRADING_BOT:
    """
    This is the constructor of NDR
    It converts and combines the given txt file for better future operation
    It also specifies the directory upon which future operation is acted
    """

    def __init__(self):
        """
        the length of the queue is 13 because previous investment need plus one and decay need to plus 5, thus 7+1+5
        """

        self.queue = deque(maxlen=13)
        self.data = pd.DataFrame()
        self.all_result = {method: [] for method in ['normalized_factor', 'decayed_alpha', 'neutral_alpha']}

    def add_day_file(self, industry_data):
        self.queue.append(industry_data)
        print(f"Added data to queue. Queue length is now {len(self.queue)}.")

    def read_and_concatenateHS300(self, base_path, start_year, end_year):
        for year in range(start_year, end_year + 1):
            for month in range(1, 3):
                for day in range(1, 32):
                    # Construct the file path for the specific day
                    day_path = os.path.join(base_path, f"{year}/{month}/{day}")

                    # Paths for each type of data expected to be found daily
                    industry_file_path = os.path.join(day_path, 'Industry.txt')
                    additional_industry_data_path = os.path.join(day_path, 'BASEDATA.txt')
                    tradable_path = os.path.join(day_path, 'SA_TRADABLESHARE_HS300.txt')

                    if os.path.exists(day_path):
                        try:
                            if os.path.exists(industry_file_path):
                                # Read the industry data
                                industry_data = pd.read_csv(industry_file_path, delimiter='|', encoding='utf-8')

                                # If there's additional industry info, merge it with industry data
                                if os.path.exists(additional_industry_data_path):
                                    additional_data = pd.read_csv(additional_industry_data_path, delimiter='|',
                                                                  encoding='utf-8')
                                    industry_data = pd.merge(industry_data, additional_data, on='SECU_CODE', how='left')

                                # Read the tradable data
                                if os.path.exists(tradable_path):
                                    tradable_data = pd.read_csv(tradable_path, delimiter='|', encoding='utf-8')

                                    # Filter the industry data for tradable stocks
                                    industry_data = industry_data[
                                        industry_data['SECU_CODE'].isin(tradable_data['SECU_CODE'])]

                                # Add to queue
                                self.add_day_file(industry_data)
                                if len(self.queue) >= 7:
                                    self.process_queue()
                                    self.calculate_adjusted_close()
                                    self.compute_ndr_factor()
                                    self.normalize_factors()
                                    self.calculate_adj_vwap()
                                    self.decay()
                                    self.industry_neutral()
                                    self.save_results(industry_data)

                        except Exception as e:
                            print(f"Error processing data for {day_path}: {e}")

    def process_queue(self):
        combined_df = pd.concat(list(self.queue), ignore_index=True)
        self.data = combined_df

    def calculate_adjusted_close(self):
        df = self.data.copy()

        if 'DIVIDEND' not in df.columns or 'SPLIT' not in df.columns:
            print("Required columns are missing.")
            return

        df.sort_values(by='TRADINGDAY_x', inplace=True)

        df['cumulative_adjustment'] = ((df['SPLIT'] + df['ACTUALPLARATIO']) * df['CLOSEPRICE']) / \
                                      (df['CLOSEPRICE'] - df['DIVIDEND'] + df['ACTUALPLARATIO'] * df['PLAPRICE'])

        df['ADJ_CLOSE_PRICE'] = df['CLOSEPRICE'] * df['cumulative_adjustment']

        self.data = df
        return df

    def compute_ndr_factor(self, days=5, delay=1):
        df = self.data.copy()
        if 'SECU_CODE' not in df.columns:
            print("Error: SECU_CODE column missing in data.")
            return

        # Ensure the data is sorted by TRADINGDAY_x within each SECU_CODE group before shifting
        df = df.sort_values(by=['SECU_CODE', 'TRADINGDAY_x'])

        # Calculate the NDR factor with the appropriate delay
        df['prev_close'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].shift(delay)
        df['prev_close_days'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].shift(delay + days)

        df['ndr_factor_notdemean'] = 1 - (df['prev_close'] / df['prev_close_days'])

        """
        ndf_factor is used for tomorrow"""
        df['ndr_factor'] = df['ndr_factor_notdemean'] - df.groupby('TRADINGDAY_x')['ndr_factor_notdemean'].transform(
            'mean')

        self.data = df

    """
    This function normalizes the alpha to get a uniform distribution within [-1, 1]
    so that the expected value of normalized ranked alpha is zero
    """

    def normalize_factors(self):
        df = self.data
        df = df.sort_values(by='TRADINGDAY_x')

        # Rank within each trading day
        df['rank'] = df.groupby('TRADINGDAY_x')['ndr_factor'].rank()

        # normalized factor for today
        df['normalized_factor'] = df.groupby('TRADINGDAY_x')['rank'].transform(
            lambda x: 2 * (x - 1) / (len(x) - 1) - 1 if len(x) > 1 else 0)

        self.data = df

    def calculate_adj_vwap(self):
        df = self.data.copy()
        # Calculate VWAP
        df['VWAP'] = df['TURNOVERVALUE'] / df['TURNOVERVOLUME']

        """
        calculate the adjusted vwap
        """
        df['ADJ_VWAP'] = df['VWAP']
        df['cumulative_vwap_adjustment'] = ((df['SPLIT'] + df['ACTUALPLARATIO']) * df['VWAP']) / \
                                           (df['VWAP'] - df['DIVIDEND'] + df['ACTUALPLARATIO'] * df['PLAPRICE'])

        # Calculate the adjusted close price (today's VWAP)
        df['ADJ_VWAP'] = df['ADJ_VWAP'] * df['cumulative_vwap_adjustment']
        self.data = df


    def decay(self):
        df = self.data.copy()
        df = df.sort_values(by=['TRADINGDAY_x', 'SECU_CODE'])
        weights = [1, 2, 3, 4, 5]
        weights = [w / sum(weights) for w in weights]
        # Apply rolling window decay within each SECU_CODE group
        df['decayed_alpha'] = df.groupby('SECU_CODE')['normalized_factor'].transform(
            lambda x: x.rolling(window=5, min_periods=1).apply(lambda y: np.dot(y, weights[-len(y):]), raw=True)
        )

        self.data = df


    """
    this function classfiy the stocks by industry and perform industry neutralization over the stocks
    this also generate another alpha for backtesting"""

    def industry_neutral(self):
        if not self.data.empty:
            try:
                df = self.data.copy()
                df = df.sort_values(by=['TRADINGDAY_x', 'SECU_CODE'])

                # Calculate average alpha for each industry category within each trading day
                df['average_alpha'] = df.groupby(['TRADINGDAY_x', 'SW2014F'])['normalized_factor'].transform('mean')
                df['neutral_alpha'] =df['normalized_factor']-df['average_alpha']

                self.data = df
                # print(df[['TRADINGDAY_x', 'SECU_CODE', 'ndr_factor', 'average_NDR', 'neutral_NDR', 'neutral_rank',
                #           'neutral_alpha']].head(100))

            except Exception as e:
                print("Failed to process industry neutralization:", e)


    def simple_backtest(self, vector, initial_capital=1e8):
        df = self.data
        df = df.sort_values(by='TRADINGDAY_x')
        if 'ADJ_CLOSE_PRICE' not in df.columns:
            print("ADJ_CLOSE_PRICE column is missing.")
            return None, None

        # Ensure there are no zero prices to avoid division by zero
        df['ADJ_CLOSE_PRICE'].replace(0, np.nan, inplace=True)
        df['ADJ_CLOSE_PRICE'].ffill(inplace=True)
        print('start')

        # Ensure vector does not contain NaNs
        vector = vector.fillna(0)

        # Append the vector to the DataFrame to align by date and stock code, assuming vector is correctly indexed
        df['vector'] = vector

        # Define masks for long and short investments based on the normalized factor
        df['long_weight'] = 0.0  # Initialize as float
        df['short_weight'] = 0.0  # Initialize as float
        df.loc[df['vector'] >= 0, 'long_weight'] = abs(df['vector']) / df[df['vector'] >= 0].groupby('TRADINGDAY_x')[
            'vector'].transform('sum').astype(float)
        df.loc[df['vector'] < 0, 'short_weight'] = abs(df['vector']) / df[df['vector'] < 0].groupby('TRADINGDAY_x')[
            'vector'].transform('sum').astype(float)

        df.loc[df['vector'] >= 0, 'weight'] = df['long_weight']
        df.loc[df['vector'] < 0, 'weight'] = df['short_weight']

        """
        all the weight is the weight of today
        """
        # Allocate capital based on weights
        df['long_capital_allocation'] = initial_capital * df['long_weight']
        df['short_capital_allocation'] = initial_capital * df['short_weight']

        # Calculate investment amount in shares for long and short positions
        df['long_investments'] = ((df['long_capital_allocation'] / df['ADJ_CLOSE_PRICE']) // 100) * 100
        df['short_investments'] = ((df['short_capital_allocation'] / df['ADJ_CLOSE_PRICE']) // 100) * 100

        # Assign investments based on the vector value
        df['investment'] = 0  # Initialize investment column with zeros

        # Assign long investments to stocks with positive vector
        df.loc[df['weight'] >= 0, 'investment'] = df['long_investments']

        # Assign short investments to stocks with negative vector
        df.loc[df['weight'] < 0, 'investment'] = df['short_investments']

        # Calculate the next-day price change
        df['next_day_return'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].diff()
        df['next_day_return'].fillna(0)  # Fill NaNs that result from diff and shift

        self.data = df
        return df

    def calculate_parameter(self, vector, initial_principal=2e8):
        df = self.data.copy()
        df.sort_values(by='TRADINGDAY_x', inplace=True)

        # Ensure that investment and price columns are numeric
        df['investment'] = pd.to_numeric(df['investment'], errors='coerce')
        df['ADJ_CLOSE_PRICE'] = pd.to_numeric(df['ADJ_CLOSE_PRICE'], errors='coerce')
        df['next_day_return'] = pd.to_numeric(df['next_day_return'], errors='coerce')

        # Shift investments to get the previous day's investments
        df['previous_investment'] = df.groupby('SECU_CODE')['investment'].shift(1)

        # Calculate investment changes
        df['investment_change'] = (df['investment'] - df['previous_investment']).fillna(0)
        df['abs_investment_change'] = df['investment_change'].abs()

        # Calculate pnl components
        df['hold_pnl'] = df['previous_investment'] * df['next_day_return']
        df['trade_pnl'] = df['investment_change'] * (
                df['ADJ_CLOSE_PRICE'] - df['ADJ_VWAP'])
        df['pnl'] = df['hold_pnl'].fillna(0) + df['trade_pnl'].fillna(0)

        df['long_pnl'] = df['pnl'] * (df['vector'] > 0)
        df['short_pnl'] = df['pnl'] * (df['vector'] <= 0)

        overall_pnl = df['pnl'].sum()

        # Calculate TVR Shares and TVR Values
        df['tvr_shares'] = df['abs_investment_change']
        df['tvr_values'] = df['abs_investment_change'] * df['ADJ_CLOSE_PRICE']

        # Ensure NaNs are handled after calculations
        df['pnl'].fillna(0, inplace=True)
        df['tvr_shares'].fillna(0, inplace=True)
        df['tvr_values'].fillna(0, inplace=True)


        self.data = df

        # Aggregate results by trading day
        pd.set_option('display.precision', 10)

        aggregated = df.groupby('TRADINGDAY_x').agg(
            pnl=('pnl', 'sum'),
            long_pnl=('long_pnl', 'sum'),
            short_pnl=('short_pnl', 'sum'),
            long_size=(
                'investment', lambda x: (x[vector.loc[x.index] >= 0] * df.loc[x.index, 'ADJ_CLOSE_PRICE']).sum()),
            short_size=(
                'investment', lambda x: (-x[vector.loc[x.index] < 0] * df.loc[x.index, 'ADJ_CLOSE_PRICE']).sum()),
            total_size=(
                'investment', lambda x: (x[vector.loc[x.index] >= 0] * df.loc[x.index, 'ADJ_CLOSE_PRICE']).sum() +
                                        (-x[vector.loc[x.index] < 0] * df.loc[x.index, 'ADJ_CLOSE_PRICE']).sum()),
            tvrshares=('tvr_shares', 'sum'),
            tvrvalues=('tvr_values', 'sum'),
            long_count=('vector', lambda x: (
                (x >= x.shift(1))).sum()),
            short_count=('vector', lambda x: (
                (x < x.shift(1))).sum()),

        ).reset_index()

        df['stocks_return'] = np.log(df['ADJ_CLOSE_PRICE'] / df['ADJ_VWAP'])

        # Calculate Information Coefficient (IC)
        aggregated['IC'] = aggregated['TRADINGDAY_x'].apply(
            lambda day: vector.corr(
                df[df['TRADINGDAY_x'] == day]['stocks_return'])
        )

        aggregated['pct_cum_pnl'] = aggregated['pnl'].cumsum() / initial_principal
        aggregated['pct_cum_long_pnl'] = aggregated['long_pnl'].cumsum() / initial_principal
        aggregated['prc_cum_short_pnl'] = aggregated['short_pnl'].cumsum() / initial_principal


        # Calculate industry distribution
        df = df.sort_values(by='TRADINGDAY_x')
        df['industry_allocation'] = (df['long_capital_allocation'] - df['short_capital_allocation']).abs()

        # Group by TRADINGDAY_x and SW2014F, then sum the industry investment
        industry_sum = df.groupby(['TRADINGDAY_x', 'SW2014F'])['industry_allocation'].sum().reset_index()

        # Calculate the percentage distribution
        industry_sum['industry_percentage'] = industry_sum['industry_allocation'] / initial_principal
        print(industry_sum['industry_percentage'])

        # Pivot the table to get the desired format
        industry_distribution = industry_sum.pivot(index='TRADINGDAY_x', columns='SW2014F',
                                                   values='industry_percentage').fillna(0)

        # Flatten the MultiIndex columns and prefix with 'industry_'
        industry_distribution.columns = [f'industry_{col}' for col in industry_distribution.columns]

        # Merge the industry distribution with the aggregated DataFrame
        aggregated = pd.merge(aggregated, industry_distribution, on='TRADINGDAY_x', how='left')


        return aggregated, overall_pnl

    def save_results(self, industry_data):
        for method in ['normalized_factor', 'decayed_alpha', 'neutral_alpha']:
            vector = self.data[method]
            self.simple_backtest(vector)
            aggregated, overall_pnl = self.calculate_parameter(vector)
            # Append the 7th day data to all_result
            self.all_result[method].append(
                aggregated[aggregated['TRADINGDAY_x'] == industry_data['TRADINGDAY_x'].max()])

        # Save results to CSV
        base_output_path = '/Users/zw/Desktop/'
        for method, result in self.all_result.items():
            all_results_df = pd.concat(result, ignore_index=True)
            output_path = os.path.join(base_output_path, f'{method}_results.csv')
            all_results_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"All results for {method} have been successfully saved to {output_path}.")


# Example usage
if __name__ == "__main__":
    analysis = NDR_TRADING_BOT()
    base_path = '/Users/zw/Desktop/5dr.data'
    analysis.read_and_concatenateHS300(base_path, 2012, 2019)
