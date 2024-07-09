import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from sklearn.linear_model import LinearRegression


class NDR:
    """
    this is the constructor of NDR
    it convert and combine the given txt file for better future operation
    it also specify the directory upon which future operation is acted
    """
    def __init__(self, file_path):
        # specify base path for file concatenation
        base_path = '/Users/zw/Desktop/5dr.data'
        start_year = 2012
        end_year = 2019
        output_csv_path = '/Users/zw/Desktop/combined_data.csv'
        
        # Get the concatenated DataFrame
        combined_data = self.read_and_concatenate(base_path, start_year, end_year)
        
        # Save the combined DataFrame to a CSV file
        self.save_data(combined_data, output_csv_path)
        self.file_path = file_path
        self.data = self.load_data()

    """
    this function read all txt file from 2012.01.01 to 2019.12.31
    select tradeable shares from all stocks in the market
    and add industry classification to each of the stocks
    """
    def read_and_concatenate(self,base_path, start_year, end_year):
        all_data_frames = []
        # Loop over each year in the specified range
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                for day in range(1, 32):
                    # Construct the file path_day specific
                    day_path = os.path.join(base_path, f"{year}/{month}/{day}")

                    # Paths for each type of data expected to be found daily
                    industry_file_path = os.path.join(day_path, 'Industry.txt')
                    additional_industry_data_path = os.path.join(day_path,
                                                                 'BASEDATA.txt')

                    if os.path.exists(day_path):
                        try:
                            if os.path.exists(industry_file_path):
                                industry_data = pd.read_csv(industry_file_path, delimiter='|', encoding='utf-8')

                                # If there's additional industry info, merge it with industry data
                                if os.path.exists(additional_industry_data_path):
                                    additional_data = pd.read_csv(additional_industry_data_path, delimiter='|',
                                                                  encoding='utf-8')
                                    industry_data = pd.merge(industry_data, additional_data, on='SECU_CODE', how='left')
                                # Read the CSV file and append to the list of DataFrames
                                all_data_frames.append(industry_data)

                        except Exception as e:
                            print(f"Error processing data for {day_path}: {e}")

        if all_data_frames:
            # Concatenate all DataFrames into a single DataFrame
            final_data_frame = pd.concat(all_data_frames, ignore_index=True)
            return final_data_frame
        else:
            print("No data found.")
            return pd.DataFrame()


    def read_and_concatenateHS300(self, base_path, start_year, end_year):
        all_data_frames = []

        # Loop over each year in the specified range
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
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

                                all_data_frames.append(industry_data)

                        except Exception as e:
                            print(f"Error processing data for {day_path}: {e}")

        # Concatenate all DataFrames
        if all_data_frames:
            concatenated_data = pd.concat(all_data_frames, ignore_index=True)
            return concatenated_data
        else:
            return pd.DataFrame()


    """
    this function save the combined data into csv file
    """
    def save_data(self,df, output_path):
        if not df.empty:
            # Save the DataFrame to a CSV file
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Data has been successfully saved to {output_path}.")
        else:
            print("No data to save.")

    """
    this function load data from csv file
    """
    def load_data(self):
        #load the data into dataframe
        if os.path.exists(self.file_path):
            try:
                df = pd.read_csv(self.file_path)
                print("Loaded data with columns:", df.columns)
                return df
            except Exception as e:
                print("Failed to read file:", e)
                return pd.DataFrame()
        else:
            print("File does not exist at:", self.file_path)
            return pd.DataFrame()

    """Calculate adjusted close prices."""

    def calculate_adjusted_close(self):
        df = self.data.copy()

        # Check for required columns
        if 'DIVIDEND' not in df.columns or 'SPLIT' not in df.columns:
            print("Required columns are missing.")
            return

        # Ensure the data is sorted by date
        df.sort_values(by='TRADINGDAY_x', inplace=True)

        # Initialize ADJ_CLOSE_PRICE with CLOSEPRICE
        """
        adj close price is shifted one backward, no need to shift again when calculating ndr factor"""



        df['cumulative_adjustment']=((df['SPLIT'] + df['ACTUALPLARATIO']) * df['CLOSEPRICE']) / \
        (df['CLOSEPRICE'] - df['DIVIDEND'] + df['ACTUALPLARATIO'] * df['PLAPRICE'])


        # Calculate the adjusted close price
        df['ADJ_CLOSE_PRICE'] = df['CLOSEPRICE'] * df['cumulative_adjustment']

        # Update the data
        self.data = df
        return df

    """
    this function compute the alpha-NDR based on prev_close and prev_close_delay
    """

    def compute_ndr_factor(self, days=5,delay=1):
        df = self.data.copy()

        if 'SECU_CODE' not in df.columns:
            print("Error: SECU_CODE column missing in data.")
            return

         # Ensure the data is sorted by TRADINGDAY_x within each SECU_CODE group before shifting
        df = df.sort_values(by=['SECU_CODE', 'TRADINGDAY_x'])

        # Calculate the NDR factor with the appropriate delay
        df['prev_close'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].shift(delay)
        df['prev_close_days'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].shift(delay+days)
        df['ndr_factor_notdemean'] = 1 - (df['prev_close'] / df['prev_close_days'])


        """
        ndf_factor is used for tomorrow"""
        df['ndr_factor'] = df['ndr_factor_notdemean'] - df.groupby('TRADINGDAY_x')['ndr_factor_notdemean'].transform('mean')

        # calculate pct_return and totalvalue for future data processing
        df = df.sort_values(by='TRADINGDAY_x')
        df['pct_return'] = np.log(df['ADJ_CLOSE_PRICE'] / df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].shift(1))
        df['TOTALVALUE'] = df['TOTALSHARES'] * df['ADJ_CLOSE_PRICE']

        self.data = df

    """
    this function normalized the alpha to get a uniform distribution within [-1,1]
    so that the expected value of normalized ranked alpha is zero"""

    def normalize_factors(self):
        df = self.data
        df = df.sort_values(by='TRADINGDAY_x')

        # Rank within each trading day
        df['rank'] = df.groupby('TRADINGDAY_x')['ndr_factor'].rank()


        # normalized factor for today
        df['normalized_factor'] = df.groupby('TRADINGDAY_x')['rank'].transform(
    lambda x: 2 * (x - 1) / (len(x) - 1) - 1 if len(x) > 1 else 0)

        self.data = df


    """
    this function perform decay5 operation on alpha and derive new alpha for backtesting
    """



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
                df['average_alpha'] = df.groupby(['TRADINGDAY_x', 'SW2014F'])['decayed_alpha'].transform('mean')
                df['neutral_alpha'] =df['decayed_alpha']-df['average_alpha']

                self.data = df


            except Exception as e:
                print("Failed to process industry neutralization:", e)


    def industry_even_neutralize(self):
        df=self.data.copy()
        df['industry_rank']=df.groupby(['SW2014F','TRADINGDAY_x'])['ndr_factor'].rank()
        df['industry_normalized_alpha']=df.groupby(['SW2014F','TRADINGDAY_x'])['industry_rank'].transform(
                    lambda x: 2 * (x - 1) / (len(x) - 1) - 1 if len(x) > 1 else 0
                )
        df['demean_industry_normalized']=df['industry_normalized_alpha']-df['average_alpha']
        self.data= df

    def industry_size_neutral(self):
        if not self.data.empty:
            try:
                df = self.data.copy()
                df = df.sort_values(by=['TRADINGDAY_x', 'SECU_CODE'])

                # Size Neutrality: Regress neutral_alpha on size within each trading day
                df['industry_size_neutral_alpha'] = np.nan
                for trading_day in df['TRADINGDAY_x'].unique():
                    daily_df = df[df['TRADINGDAY_x'] == trading_day].copy()
                    X = np.log(daily_df['TOTALVALUE']).values.reshape(-1, 1)  # Log of size (market value)
                    y = daily_df['neutral_alpha'].fillna(0).values
                    if len(y) > 1:  # Perform regression only if there are enough data points
                        reg = LinearRegression().fit(X, y)
                        df.loc[df['TRADINGDAY_x'] == trading_day, 'industry_size_neutral_alpha'] = y - reg.predict(X)

                self.data = df

            except Exception as e:
                print("Failed to process industry size neutralization:", e)
        else:
            print('empty dataframe')



    def mark_limit_and_suspension_conditions(self):
        df = self.data
        df= df.sort_values(by='TRADINGDAY_x')
        df['IS_LIMIT_UP'] = df.groupby('SECU_CODE')['pct_return'].shift(1) >= 0.10
        df['IS_LIMIT_DOWN'] = df.groupby('SECU_CODE')['pct_return'].shift(1) <=-0.10
        df['IS_SUSPENDED'] = df['pct_return'].isna()
        self.data = df

    def winsorized_factor(self, lower_quantile=0.01, upper_quantile=0.99):
        if not self.data.empty:
            try:
                df = self.data.copy()
                if 'TRADINGDAY_x' in df.index.names:
                    df = df.reset_index()  # Reset index if TRADINGDAY_x is part of the index
                df = df.sort_values(by='TRADINGDAY_x')

                # Ensure limit conditions and suspensions are marked
                if 'IS_LIMIT_UP' not in df.columns or 'IS_LIMIT_DOWN' not in df.columns or 'IS_SUSPENDED' not in df.columns:
                    print(
                        "Limit conditions or suspensions not marked. Please run mark_limit_and_suspension_conditions() first.")
                    return

                # Calculate median for each trading day
                df['daily_median'] = df.groupby('TRADINGDAY_x')['industry_size_neutral_alpha'].transform('median').abs()

                # Set factor to median for limit up, limit down, or suspended stocks
                df.loc[df['IS_LIMIT_UP'], 'industry_size_neutral_alpha'] = -1*df['daily_median']

                df.loc[df['IS_LIMIT_DOWN'], 'industry_size_neutral_alpha'] = df['daily_median']

                df.loc[df['IS_SUSPENDED'], 'industry_size_neutral_alpha'] = 0


                # Apply limits within each trading day
                def apply_limits(group):
                    lower_limit = group['industry_size_neutral_alpha'].quantile(lower_quantile)
                    upper_limit = group['industry_size_neutral_alpha'].quantile(upper_quantile)
                    median = group['daily_median'].iloc[0]
                    print(median)
                    group['winsorized_factor'] = np.where(group['industry_size_neutral_alpha'] < lower_limit,lower_limit,
                                                          np.where(group['industry_size_neutral_alpha'] > upper_limit, median,
                                                                   group['industry_size_neutral_alpha']))
                    return group

                df = df.groupby('TRADINGDAY_x').apply(apply_limits).reset_index(drop=True)
                self.data = df

            except Exception as e:
                print('Winsorization failed:', e)
        else:
            print('Empty dataframe')

    """
    this function perform backtesting over the three alpha derived above
    the initial principal is always 2e8, and pnl records the profit/loss from day di-1 to day di
    ex-right and backward, forward adjustment is also taken into account
    trading share is always multiple of 100"""


    def calculate_adj_vwap(self):
        df=self.data.copy()
        # Calculate VWAP
        # df['VOLUME'] = df.groupby('SECU_CODE')['TURNOVERVALUE'].shift(1) / df['ADJ_CLOSE_PRICE']
        df['VWAP'] = df['TURNOVERVALUE'] / df['TURNOVERVOLUME']

        """
        calculate the adjusted vwap"""

        df['ADJ_VWAP'] = df['VWAP']
        df['cumulative_vwap_adjustment'] = ((df['SPLIT'] + df['ACTUALPLARATIO']) * df[
            'VWAP']) / \
                                           (df['VWAP'] - df['DIVIDEND'] + df[
                                               'ACTUALPLARATIO'] * df['PLAPRICE'])

        # Calculate the adjusted close price(today's vwap)
        df['ADJ_VWAP'] = df['ADJ_VWAP'] * df['cumulative_vwap_adjustment']
        self.data=df

    def simple_backtest(self, vector, initial_capital=1e8):
        df = self.data
        df = df.sort_values(by='TRADINGDAY_x')
        if 'ADJ_CLOSE_PRICE' not in df.columns:
            print("ADJ_CLOSE_PRICE column is missing.")
            return None, None

        # Ensure there are no zero prices to avoid division by zero
        df['ADJ_CLOSE_PRICE'].replace(0, np.nan, inplace=True)
        df['ADJ_CLOSE_PRICE'].ffill(inplace=True)

        # Ensure vector does not contain NaNs
        vector = vector.fillna(0)

        # Append the vector to the DataFrame to align by date and stock code, assuming vector is correctly indexed
        df['vector'] = vector

        def weight_assignment(df, vector):
            df['vector'] = vector
            df = df.sort_values(by='TRADINGDAY_x')

            # Define masks for long and short investments based on the normalized factor
            df['long_weight'] = 0.0
            df['short_weight'] = 0.0

            df.loc[df['vector'] >= 0, 'long_weight'] = abs(df['vector']) / \
                                                       df[df['vector'] >= 0].groupby('TRADINGDAY_x')[
                                                           'vector'].transform('sum')
            df.loc[df['vector'] < 0, 'short_weight'] = abs(df['vector']) / \
                                                       df[df['vector'] < 0].groupby('TRADINGDAY_x')['vector'].transform(
                                                           'sum')

            df.loc[df['vector'] >= 0, 'weight'] = df['long_weight']
            df.loc[df['vector'] < 0, 'weight'] = df['short_weight']
            return df


        def weight_assignment_industry_uniform(df, vector):
            df = df.sort_values(by='TRADINGDAY_x')
            num_industries = df.groupby('TRADINGDAY_x')['SW2014F'].nunique().astype('float64')

            # Merge num_industries back into the original dataframe to align with TRADINGDAY_x
            df = df.merge(num_industries, on='TRADINGDAY_x', suffixes=('', '_count'))

            # Calculate the industry weight and ensure no NaN values
            df['industry_weight'] = 1 / df['SW2014F_count']

            df['vector'] = vector
            df['long_weight'] = 0.0
            df['short_weight'] = 0.0

            # Calculate long_sum and short_sum grouped by industry
            df['long_sum'] = df[df['vector'] >= 0.2].groupby(['TRADINGDAY_x', 'SW2014F'])['vector'].transform('sum')
            # Filter the DataFrame to include only rows where 'vector' is less than 0.6
            # Group by 'TRADINGDAY_x' and 'SW2014F', and calculate the sum of the absolute values of 'vector'
            df['short_sum']= -1*df[df['vector'] < 0.2].groupby(['TRADINGDAY_x', 'SW2014F'])['vector'].transform(lambda x: x.abs().sum())




            # Initialize weights
            df['long_weight'] = 0.0
            df['short_weight'] = 0.0

            # Explicitly cast the columns to float64 before assignment
            df['industry_weight'] = df['industry_weight'].astype('float64')
            df['vector'] = df['vector'].astype('float64')
            df['long_sum'] = df['long_sum'].astype('float64')
            df['short_sum'] = df['short_sum'].astype('float64')
            df['long_weight'] = df['long_weight'].astype('float64')
            df['short_weight'] = df['short_weight'].astype('float64')

            df.loc[df['vector'] >= 0.2, 'long_weight'] = (df['industry_weight'] * df['vector']).abs() / df['long_sum']
            df.loc[df['vector'] < 0.2, 'short_weight'] = (df['industry_weight'] * df['vector']).abs() / df['short_sum']

            df.loc[df['vector'] >= 0.2, 'weight'] = df['long_weight']
            df.loc[df['vector'] < 0.2, 'weight'] = df['short_weight']

            return df

        def assign_weights(df, vector):
            if vector.name == 'demean_industry_normalized':
                df = weight_assignment_industry_uniform(df, vector)
            else:
                df = weight_assignment(df, vector)

            df.loc[df['vector'] >= 0, 'weight'] = df['long_weight']
            df.loc[df['vector'] < 0, 'weight'] = df['short_weight']

            return df
        df= assign_weights(df,vector)


        # Allocate capital based on weights
        df['long_capital_allocation'] = initial_capital * df['long_weight']
        df['short_capital_allocation'] = initial_capital * df['short_weight']

        # Calculate investment amount in shares for long and short positions
        # round to the closest multiple of 100 smaller than the original number
        df['long_investments'] = ((df['long_capital_allocation'] / df['ADJ_CLOSE_PRICE']) // 100) * 100
        df['short_investments'] = ((df['short_capital_allocation'] / df['ADJ_CLOSE_PRICE']) // 100) * 100

        # Assign investments based on the vector value
        df['investment'] = 0  # Initialize investment column with zeros

        # Assign long investments to stocks with positive vector
        df.loc[df['weight'] >= 0, 'investment'] = df['long_investments']

        # Assign short investments to stocks with negative vector
        df.loc[df['weight'] < 0, 'investment'] = df['short_investments']

        # """
        # right now the investment column is the investment happened within that day"""
        #


        # Calculate the next-day price change
        """
        notice that adj close is now prev close
        when calculate the diff, it is only the diff between today and yesterday"""

        df['next_day_return'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].diff()
        df['next_day_return'].fillna(0)  # Fill NaNs that result from diff and shift

        self.data = df
        return df


    def calculate_parameter(self, vector,initial_principal=2e8):
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
        # Define the condition
        """
        notice that we hold only when the sign of pervious investment and current investment is the same
        """
        condition = df['previous_investment'] * df['investment'] > 0

        # Calculate hold_pnl based on the condition
        df['hold_pnl'] = np.where(condition, df['previous_investment'] * df['next_day_return'], 0)

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

        # Calculate annualized returns
        df['year'] = df['TRADINGDAY_x'].astype(str).str[:4]
        df['annual_pnl'] =  df.groupby('year')['pnl'].transform('sum')
        df['annualized_return'] = df['annual_pnl'] / initial_principal


        self.data = df

        # Aggregate results by trading day
        pd.set_option('display.precision', 10)



        aggregated = df.groupby('TRADINGDAY_x').agg(
            pnl=('pnl', 'sum'),
            long_pnl=('long_pnl','sum'),
            short_pnl=('short_pnl','sum'),
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
                        (x>=x.shift(1))).sum()),
            short_count=('vector', lambda x: (
                        (x<x.shift(1))).sum()),
            annualized_return=('annualized_return','first')
        ).reset_index()
        print(aggregated['annualized_return'])


        df['stocks_return'] = np.log(df['ADJ_CLOSE_PRICE']/df['ADJ_VWAP'])

        # Calculate Information Coefficient (IC)
        aggregated['IC'] = aggregated['TRADINGDAY_x'].apply(
            lambda day: vector.corr(
                df[df['TRADINGDAY_x'] == day]['stocks_return'])
        )

        aggregated['pct_cum_pnl']= aggregated['pnl'].cumsum()/initial_principal
        aggregated['pct_cum_long_pnl']=aggregated['long_pnl'].cumsum()/initial_principal
        aggregated['pct_cum_short_pnl']=aggregated['short_pnl'].cumsum()/initial_principal


        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        def plot_combined_graphs(aggregated, df, initial_principal, vector):
            # Ensure TRADINGDAY_x is treated as datetime
            aggregated['TRADINGDAY_x'] = pd.to_datetime(aggregated['TRADINGDAY_x'], format='%Y%m%d')
            df['TRADINGDAY_x'] = pd.to_datetime(df['TRADINGDAY_x'], format='%Y%m%d')

            cumulative_avg_return = df.groupby('TRADINGDAY_x')['pct_return'].mean().cumsum()

            # Calculate TVR ratio
            aggregated['tvr_ratio'] = aggregated['tvrvalues'] / initial_principal

            # Calculate excess returns
            aggregated[f'{vector.name}_excess_pnl'] = aggregated['pct_cum_pnl'] - cumulative_avg_return.reindex(
                aggregated['TRADINGDAY_x']).values
            aggregated[f'{vector.name}_excess_long_pnl'] = aggregated['pct_cum_long_pnl'] - cumulative_avg_return.reindex(
                aggregated['TRADINGDAY_x']).values
            aggregated[f'{vector.name}_excess_short_pnl'] = aggregated['pct_cum_short_pnl'] - cumulative_avg_return.reindex(
                aggregated['TRADINGDAY_x']).values

            fig, axs = plt.subplots(3, 1, figsize=(10, 8))

            # Plot cumulative PnL
            axs[0].plot(aggregated['TRADINGDAY_x'], aggregated['pct_cum_pnl'], label='Cumulative PnL')
            axs[0].plot(aggregated['TRADINGDAY_x'], aggregated['pct_cum_long_pnl'], label='Cumulative Long PnL')
            axs[0].plot(aggregated['TRADINGDAY_x'], aggregated['pct_cum_short_pnl'], label='Cumulative Short PnL')
            axs[0].plot(cumulative_avg_return.index, cumulative_avg_return.values, label='Cumulative Average Return')
            axs[0].xaxis.set_major_locator(mdates.YearLocator())
            axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axs[0].set_title('Cumulative PnL, Long PnL, Short PnL, and Cumulative Average Return', fontsize='small')
            axs[0].set_xlabel('Trading Day', fontsize='small')
            axs[0].set_ylabel('Cumulative Return', fontsize='small')
            axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            axs[0].grid(True)

            # Plot histogram of TVR ratio
            axs[1].hist(aggregated['tvr_ratio'], bins=30, color='blue', edgecolor='black', alpha=0.7)
            axs[1].set_title('Distribution of TVR Ratio', fontsize='small')
            axs[1].set_xlabel('TVR Ratio', fontsize='small')
            axs[1].set_ylabel('Frequency', fontsize='small')
            axs[1].grid(True)

            # Plot annualized return
            axs[2].plot(aggregated['TRADINGDAY_x'], aggregated['annualized_return'], label='Annualized Return')
            axs[2].xaxis.set_major_locator(mdates.YearLocator())
            axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axs[2].set_title('Annualized Return Over Time', fontsize='small')
            axs[2].set_xlabel('Trading Day', fontsize='small')
            axs[2].set_ylabel('Annualized Return', fontsize='small')
            axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            axs[2].grid(True)

            fig, axs = plt.subplots(2, 1, figsize=(14, 8))
            # Plot excess returns
            axs[0].plot(aggregated['TRADINGDAY_x'], aggregated[f'{vector.name}_excess_pnl'], label='Excess PnL')
            axs[0].plot(aggregated['TRADINGDAY_x'], aggregated[f'{vector.name}_excess_long_pnl'], label='Excess Long PnL')
            axs[0].plot(aggregated['TRADINGDAY_x'], aggregated[f'{vector.name}_excess_short_pnl'], label='Excess Short PnL')
            axs[0].xaxis.set_major_locator(mdates.YearLocator())
            axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axs[0].set_title('Excess Returns (Overall, Long, Short)', fontsize='small')
            axs[0].set_xlabel('Trading Day', fontsize='small')
            axs[0].set_ylabel('Excess Return', fontsize='small')
            axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            axs[0].grid(True)

            # Filter for the year 2019
            df_2019 = df[df['TRADINGDAY_x'].dt.year == 2019]
            aggregated_2019 = aggregated[aggregated['TRADINGDAY_x'].dt.year == 2019]
            cumulative_avg_return_2019 = df_2019.groupby('TRADINGDAY_x')['pct_return'].mean().cumsum()

            # Calculate 2019 excess returns
            aggregated_2019[f'{vector.name}_excess_pnl'] = aggregated_2019['pnl'].transform('cumsum')/initial_principal - cumulative_avg_return_2019.reindex(
                aggregated_2019['TRADINGDAY_x']).values
            aggregated_2019[f'{vector.name}_excess_long_pnl'] = aggregated_2019[
                                                     'long_pnl'].transform('cumsum')/initial_principal - cumulative_avg_return_2019.reindex(
                aggregated_2019['TRADINGDAY_x']).values
            aggregated_2019[f'{vector.name}_excess_short_pnl'] = aggregated_2019[
                                                      'short_pnl'].transform('cumsum')/initial_principal - cumulative_avg_return_2019.reindex(
                aggregated_2019['TRADINGDAY_x']).values

            # Plot 2019 excess returns
            axs[1].plot(aggregated_2019['TRADINGDAY_x'], aggregated_2019[f'{vector.name}_excess_pnl'], label='Excess PnL 2019')
            axs[1].plot(aggregated_2019['TRADINGDAY_x'], aggregated_2019[f'{vector.name}_excess_long_pnl'],
                        label='Excess Long PnL 2019')
            axs[1].plot(aggregated_2019['TRADINGDAY_x'], aggregated_2019[f'{vector.name}_excess_short_pnl'],
                        label='Excess Short PnL 2019')
            axs[1].xaxis.set_major_locator(mdates.MonthLocator())
            axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axs[1].set_title('Excess Returns (Overall, Long, Short) in 2019', fontsize='small')
            axs[1].set_xlabel('Trading Day', fontsize='small')
            axs[1].set_ylabel('Excess Return', fontsize='small')
            axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            axs[1].grid(True)


            plt.tight_layout()
            plt.show()


        plot_combined_graphs(aggregated,df,initial_principal,vector)



        # Calculate Sharpe Ratio
        daily_returns = (aggregated['pnl']/initial_principal).fillna(0)
        sharpe_ratio = np.sqrt(252)*daily_returns.mean() / daily_returns.std()
        aggregated['sharpe_ratio'] = sharpe_ratio



        def calculate_max_drawdown (df):
            df = df.sort_values(by='TRADINGDAY_x')  # Ensure data is sorted by date
            max_drawdown = 0
            for i in range(1, len(df)):
                drawdown= aggregated.loc[i,'pnl']/initial_principal
                if drawdown<max_drawdown:
                    max_drawdown=drawdown
                df.loc[i,'mdd'] = max_drawdown


            return df

        def cal_industry_distribution(df, aggregated, initial_principal):
            # Calculate industry distribution
            df = df.sort_values(by='TRADINGDAY_x')
            df['net_allocation'] = (df['long_capital_allocation'] - df['short_capital_allocation']).abs()

            # Group by TRADINGDAY_x and SW2014F, then sum the industry investment
            industry_sum = df.groupby(['TRADINGDAY_x', 'SW2014F'])['net_allocation'].sum().reset_index()

            # Calculate the percentage distribution
            industry_sum['industry_percentage'] = industry_sum['net_allocation'] / initial_principal

            # Pivot the table to get the desired format
            industry_distribution = industry_sum.pivot(index='TRADINGDAY_x', columns='SW2014F',
                                                       values='industry_percentage').fillna(0)

            # Flatten the MultiIndex columns and prefix with 'industry_'
            industry_distribution.columns = [f'industry_{col}' for col in industry_distribution.columns]

            # Merge the industry distribution with the aggregated DataFrame
            aggregated = pd.merge(aggregated, industry_distribution, on='TRADINGDAY_x', how='left')

            # Calculate the average industry percentage in each industry over trading day
            average_industry_percentage = industry_sum.groupby('SW2014F')['industry_percentage'].mean()

            return aggregated, average_industry_percentage

        def grouping_analysis(df):
            # Sort the dataframe by vector and trading day for proper quantile grouping
            df = df.sort_values(by=['TRADINGDAY_x', 'vector'])

            # Initialize a list to store average returns per day
            average_returns_per_day_list = []
            average_size_per_day_list = []


            # Loop over each trading day to calculate daily average returns for each vector group
            for trading_day in df['TRADINGDAY_x'].unique():
                daily_df = df[df['TRADINGDAY_x'] == trading_day].copy()
                daily_df['vector_group'] = pd.qcut(daily_df['vector'], q=50, labels=False, duplicates='drop')
                daily_average_returns = daily_df.groupby('vector_group')['pct_return'].mean().reset_index()
                daily_average_size = daily_df.groupby('vector_group')['TOTALVALUE'].mean().reset_index()
                daily_average_returns['TRADINGDAY_x'] = trading_day
                daily_average_size['TRADINGDAY_x'] = trading_day
                average_returns_per_day_list.append(daily_average_returns)
                average_size_per_day_list.append(daily_average_size)

            # Concatenate the daily average returns into a single dataframe
            average_returns_per_day = pd.concat(average_returns_per_day_list)
            average_size_per_day = pd.concat(average_size_per_day_list)
            average_returns_per_day.set_index('TRADINGDAY_x', inplace=True)
            average_size_per_day.set_index('TRADINGDAY_x', inplace=True)

            # Calculate the overall average return and size for each vector group across all days
            average_returns = average_returns_per_day.groupby('vector_group')['pct_return'].mean().reset_index()
            average_size = average_size_per_day.groupby('vector_group')['TOTALVALUE'].mean().reset_index()

            # Rename columns for clarity
            average_returns.columns = ['vector_group', 'average_return']
            average_returns = average_returns.sort_values(by='vector_group', ascending=True)
            average_size.columns = ['vector_group', 'average_size']
            average_size = average_size.sort_values(by='vector_group', ascending=True)

            # Create subplots
            fig, axs = plt.subplots(2, 1, figsize=(14, 10))

            # Plotting average returns
            axs[0].bar(average_returns['vector_group'], average_returns['average_return'], color='b', alpha=0.6)
            axs[0].set_xlabel('Vector Group', fontsize='small')
            axs[0].set_ylabel('Average Return', fontsize='small')
            axs[0].set_title('Average Return by Vector Group', fontsize='small')
            axs[0].grid(True)

            # Plotting average sizes
            axs[1].bar(average_size['vector_group'], average_size['average_size'], color='r', alpha=0.6)
            axs[1].set_xlabel('Vector Group', fontsize='small')
            axs[1].set_ylabel('Average Size', fontsize='small')
            axs[1].set_title('Average Size by Vector Group', fontsize='small')
            axs[1].grid(True)

            plt.tight_layout()  # Add more space between plots
            plt.show()

        aggregated= calculate_max_drawdown(aggregated)
        aggregated, average_industry_percentage= cal_industry_distribution(df,aggregated,initial_principal)
        grouping_analysis(df)

        """
        step1 统计 a. 行业过去5/20天的累计收益排名情况 b. 5dr/20dr因子的选股 尝试比如5/10分组，查看a和b两者之间有无关联？
        step2 5dr/20dr因子选股的市值暴露情况如何 （比如将因子值分n组，统计每组内的选股的市值均值）
        step3 尝试对5dr/20dr做市值&行业中性，得到insn版本(industry neutral size neutral的简称)的因子 这个因子的分组暴露情况又是如何？
        """

        def industry_pnl_stats(df, aggregated, vector, start_date='2019-12-02', end_date='2019-12-31'):
            # Ensure 'TRADINGDAY_x' is treated as datetime
            df['TRADINGDAY_x'] = pd.to_datetime(df['TRADINGDAY_x'], format='%Y%m%d')
            aggregated['TRADINGDAY_x'] = pd.to_datetime(aggregated['TRADINGDAY_x'], format='%Y%m%d')

            # Calculate net capital allocation
            df['net_allocation'] = df['long_capital_allocation'] - df['short_capital_allocation']
            df['vector'] = vector

            # Filter the dataframe for the given date range
            recent_df = df[(df['TRADINGDAY_x'] >= start_date) & (df['TRADINGDAY_x'] <= end_date)]

            if recent_df.empty:
                print(f"No data found in the date range {start_date} to {end_date}")
                return aggregated

            # Sort the dataframe by 'TRADINGDAY_x'
            recent_df = recent_df.sort_values(by='TRADINGDAY_x')

            if 'TOTALVALUE' not in recent_df.columns or recent_df['TOTALVALUE'].isnull().all():
                print("Column 'TOTALVALUE' is missing or all values are NaN in the recent_df.")
                return aggregated

            # Ensure no NaNs in 'TOTALVALUE' before quantile calculation
            recent_df = recent_df.dropna(subset=['TOTALVALUE'])

            # Check if recent_df has sufficient data for quantile calculation
            if len(recent_df) < 10:
                print("Not enough data for quantile calculation in recent_df.")
                return aggregated

            # Group by 'SW2014F' and 'TRADINGDAY_x' and calculate sum of 'pct_return' for each group
            recent_df['size_group'] = pd.qcut(recent_df['TOTALVALUE'], q=10, labels=False, duplicates='drop')

            grouped = recent_df.groupby(['TRADINGDAY_x', 'SW2014F'])['pct_return'].sum().reset_index()
            size_grouped = recent_df.groupby(['TRADINGDAY_x', 'size_group'])['pct_return'].sum().reset_index()

            # Calculate the cumulative sum of 'pct_return' for each industry and size group
            grouped['cumulative_return'] = grouped.groupby('SW2014F')['pct_return'].cumsum()
            size_grouped['cumulative_return'] = size_grouped.groupby('size_group')['pct_return'].cumsum()

            # Calculate the percentage contribution of each industry and size group using the cumulative sum
            grouped['industry_contribution'] = grouped['cumulative_return']
            size_grouped['size_contribution'] = size_grouped['cumulative_return']

            # Pivot the table to have 'TRADINGDAY_x' as index and 'SW2014F' as columns
            pivot_df_industry = grouped.pivot(index='TRADINGDAY_x', columns='SW2014F', values='industry_contribution')
            pivot_df_size = size_grouped.pivot(index='TRADINGDAY_x', columns='size_group', values='size_contribution')

            # Merge the pivot dataframe with the aggregated dataframe on 'TRADINGDAY_x'
            aggregated = pd.merge(aggregated, pivot_df_industry, on='TRADINGDAY_x', how='left')

            # Sort the dataframe by 'TRADINGDAY_x'
            recent_df = recent_df.sort_values(by='TRADINGDAY_x')

            # Add vector group information
            recent_df['vector_group'] = pd.qcut(recent_df['vector'], q=10, labels=False, duplicates='drop')

            # Calculate the average industry exposure within each vector group
            industry_exposure = recent_df.groupby(['vector_group', 'SW2014F'])['net_allocation'].sum().reset_index()
            total_allocation_per_group = recent_df.groupby('vector_group')['net_allocation'].sum().reset_index()
            industry_exposure = industry_exposure.merge(total_allocation_per_group, on='vector_group',
                                                        suffixes=('', '_group_total'))

            # The percentage is the allocation of capital within vector group within industry
            industry_exposure['allocation_percentage'] = industry_exposure['net_allocation'] / industry_exposure[
                'net_allocation_group_total']

            # Pivot the table to have 'vector_group' as index and 'SW2014F' as columns
            pivot_df_industry_exposure = industry_exposure.pivot(index='vector_group', columns='SW2014F',
                                                                 values='allocation_percentage').fillna(0)

            # Calculate the average size exposure within each vector group
            size_exposure = recent_df.groupby(['vector_group', 'size_group'])['net_allocation'].sum().reset_index()
            total_allocation_per_size_group = recent_df.groupby('vector_group')['net_allocation'].sum().reset_index()
            size_exposure = size_exposure.merge(total_allocation_per_size_group, on='vector_group',
                                                suffixes=('', '_group_total'))

            # The percentage is the allocation of capital within vector group within size group
            size_exposure['allocation_percentage'] = size_exposure['net_allocation'] / size_exposure[
                'net_allocation_group_total']

            # Pivot the table to have 'vector_group' as index and 'size_group' as columns
            pivot_df_size_exposure = size_exposure.pivot(index='vector_group', columns='size_group',
                                                         values='allocation_percentage').fillna(0)

            # Plotting return contributions
            fig, axs = plt.subplots(2, 1, figsize=(14, 10))

            # Plotting the industry return data
            pivot_df_industry.plot(kind='bar', stacked=True, ax=axs[0])
            axs[0].set_title('Industry Contribution to Return Over 20 Days Ending on 2019-12-31', fontsize='small')
            axs[0].set_ylabel('Cumulative Contribution', fontsize='small')
            axs[0].set_xlabel('Trading Day', fontsize='small')
            axs[0].legend(title='Industry', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

            # Plotting the size return data
            pivot_df_size.plot(kind='bar', stacked=True, ax=axs[1])
            axs[1].set_title('Size Group Contribution to Return Over 20 Days Ending on 2019-12-31', fontsize='small')
            axs[1].set_ylabel('Cumulative Contribution', fontsize='small')
            axs[1].set_xlabel('Trading Day', fontsize='small')
            axs[1].legend(title='Size Group', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

            plt.tight_layout()
            plt.show()

            # Plotting exposure percentages
            fig, axs = plt.subplots(2, 1, figsize=(14, 10))

            # Plotting the industry exposure percentages
            pivot_df_industry_exposure.plot(kind='bar', stacked=True, ax=axs[0])
            axs[0].set_title('Average Industry Exposure as Percentage of Total Allocation by Vector Group', fontsize='small')
            axs[0].set_ylabel('Exposure Percentage', fontsize='small')
            axs[0].set_xlabel('Vector Group', fontsize='small')
            axs[0].legend(title='Industry', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

            # Plotting the size exposure percentages
            pivot_df_size_exposure.plot(kind='bar', stacked=True, ax=axs[1])
            axs[1].set_title('Average Size Exposure as Percentage of Total Allocation by Vector Group', fontsize='small')
            axs[1].set_ylabel('Exposure Percentage', fontsize='small')
            axs[1].set_xlabel('Vector Group', fontsize='small')
            axs[1].legend(title='Size Group', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

            plt.tight_layout()
            plt.show()

            return aggregated

        aggregated= industry_pnl_stats(df, aggregated, vector)

        def plot_recent_performance(stats):
            """
            Plots recent performance metrics based on the provided stats and factor data.

            Parameters:
            stats (DataFrame): DataFrame containing various statistics including normalized alpha, neutral alpha, and next day returns.
            """

            # Extract relevant data from stats DataFrame
            factor_data = stats['normalized_factor']
            neutral_factor_data = stats['industry_size_neutral_alpha']

            # Initialize subplots
            fig, axs = plt.subplots(3, 1, figsize=(14, 8))

            # 3.3 Plot distribution of raw and neutral factor values
            ax1, ax2 = axs[0], axs[0].twinx()

            # Plot histograms
            factor_data.hist(bins=50, ax=ax1, alpha=0.7, color='blue')
            neutral_factor_data.hist(bins=50, ax=ax2, alpha=0.7, color='red')

            # Set labels and title
            ax1.set_xlabel('Factor Value')
            ax1.set_ylabel('Raw Factor Distribution', color='blue', fontsize='small')
            ax2.set_ylabel('Neutral Factor Distribution', color='red', fontsize='small')
            ax1.set_title('Distribution of Raw and Neutral Factors', fontsize='small')

            # Create legends
            ax1_lines, ax1_labels = ax1.get_legend_handles_labels()
            ax2_lines, ax2_labels = ax2.get_legend_handles_labels()

            # Add legends to the plot
            ax1.legend(ax1_lines, ['Raw Factor'], loc='upper left', fontsize='small')
            ax2.legend(ax2_lines, ['Neutral Factor'], loc='upper right', fontsize='small')

            def calculate_ic_means(factor_data, neutral_factor_data, stats, max_delay=30):
                delays = range(1, max_delay + 1)
                raw_ic_means = []
                neutral_ic_means = []

                for delay in delays:
                    shifted_returns = stats.groupby('SECU_CODE')['next_day_return'].shift(-delay)

                    raw_ic = factor_data.corr(shifted_returns)
                    neutral_ic = neutral_factor_data.corr(shifted_returns)

                    raw_ic_means.append(raw_ic)
                    neutral_ic_means.append(neutral_ic)

                return delays, raw_ic_means, neutral_ic_means

            # Plot IC mean values for different delays
            def plot_ic_means(delays, raw_ic_means, neutral_ic_means, ax):
                ax.plot(delays, raw_ic_means, label='Raw IC Mean', color='blue')
                ax.plot(delays, neutral_ic_means, label='Neutral IC Mean', color='red')

                ax.set_title('IC Mean Values for Different Delays', fontsize='small')
                ax.set_xlabel('Delay (days)', fontsize='small')
                ax.set_ylabel('IC Mean', fontsize='small')
                ax.legend(fontsize='small')

            # Calculate IC means
            delays, raw_ic_means, neutral_ic_means = calculate_ic_means(factor_data, neutral_factor_data, stats)

            # Plot IC means
            plot_ic_means(delays, raw_ic_means, neutral_ic_means, axs[1])

            # 3.9 Plot average daily number of stocks selected by factor across industries
            daily_stock_counts_raw = stats.groupby(['TRADINGDAY_x', 'SW2014F']).apply(
                lambda x: (factor_data.loc[x.index] != 0).sum())
            daily_stock_counts_neutral = stats.groupby(['TRADINGDAY_x', 'SW2014F']).apply(
                lambda x: (neutral_factor_data.loc[x.index] != 0).sum())
            daily_avg_stock_counts_raw = daily_stock_counts_raw.groupby('SW2014F').mean()
            daily_avg_stock_counts_neutral = daily_stock_counts_neutral.groupby('SW2014F').mean()

            # Plot average daily number of stocks selected by factor across industries
            daily_avg_stock_counts_raw.plot(kind='line', ax=axs[2], label='Raw Factor')
            daily_avg_stock_counts_neutral.plot(kind='line', ax=axs[2], label='Neutral Factor')

            axs[2].set_title('Average Daily Number of Stocks Selected by Factor Across Industries')
            axs[2].set_xlabel('Industry')
            axs[2].set_ylabel('Average Number of Stocks')
            axs[2].legend(fontsize='small')

            plt.tight_layout()
            plt.show()

        # Example usage
        plot_recent_performance(df)

        return aggregated,overall_pnl


    def load_index_data(self, df):
        index_path = '/Users/zw/Desktop/沪深300指数历史数据 (1).csv'  # specify the path to your index data file
        index = pd.read_csv(index_path)

        # Convert the '日期' column to datetime format
        index['日期'] = pd.to_datetime(index['日期'])

        # Filter the index data to the specified date range
        start_date = '2012-01-04'
        end_date = '2019-12-31'
        index_filtered = index[(index['日期'] >= start_date) & (index['日期'] <= end_date)]

        # Convert '日期' to the format YYYYMMDD
        index_filtered['TRADINGDAY_x'] = index_filtered.sort_index(ascending=False).reset_index(drop=True)['日期'].dt.strftime('%Y%m%d')

        # Ensure 'TRADINGDAY_x' is in string format for merging
        index_filtered['TRADINGDAY_x'] = index_filtered['TRADINGDAY_x'].astype(int)

        # Select only the necessary columns for merging
        index_filtered = index_filtered[['TRADINGDAY_x', '收盘']].rename(columns={'收盘': 'index_close'})

        # Convert 'index_close' to numeric using astype
        index_filtered['index_close'] = index_filtered['index_close'].str.replace(',', '').astype(float)

        # Convert 'index_close' to numeric, forcing errors to NaN and then fill them
        index_filtered['index_close'] = pd.to_numeric(index_filtered['index_close'], errors='coerce').fillna(
            method='ffill')

        # Calculate index returns
        index_filtered['index_return'] = index_filtered['index_close'].pct_change().fillna(0)



        # Calculate cumulative returns
        index_filtered['cumulative_index_return'] = (1 + index_filtered['index_return']).cumprod() - 1

        # Sort the index data by 'TRADINGDAY_x' to match the original DataFrame
        index_filtered = index_filtered.sort_values(by='TRADINGDAY_x')

        # Merge with the input DataFrame on 'TRADINGDAY_x'
        df['TRADINGDAY_x'] = df['TRADINGDAY_x'].astype(int)
        df = pd.merge(df, index_filtered, on='TRADINGDAY_x', how='left')

        # Sort the merged DataFrame by 'TRADINGDAY_x'
        df = df.sort_values(by='TRADINGDAY_x')

        # Save the index return data to a CSV file
        index_df = df[['TRADINGDAY_x', 'index_close', 'index_return', 'cumulative_index_return']].copy()
        index_df.to_csv('/Users/zw/Desktop/玄元投资实习/玄元TASK1/index_return.csv', index=False, encoding='utf-8-sig')
        print('Index return saved to /Users/zw/Desktop/玄元投资实习/玄元TASK1/index_return.csv')
        return df

    def plot_pnl(self, df, method_name):
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df.sort_values(by='TRADINGDAY_x')['pct_pnl'], label='Overall P&L', color='blue')
        plt.plot(df.index, df.sort_values(by='TRADINGDAY_x')['cumulative_index_return'], label='HS300 P&L', color='black')
        plt.plot(df.index, df.sort_values(by='TRADINGDAY_x')['pct_cumulative_long'], label='Long P&L', color='green')
        plt.plot(df.index, df.sort_values(by='TRADINGDAY_x')['pct_cumulative_short'], label='Short P&L', color='red')
        plt.title(f'Cumulative P&L Over Time ({method_name})')
        plt.xlabel('Trading Day')
        plt.ylabel('Cumulative P&L')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


    def run_analysis(self):
        if not self.data.empty:
            self.calculate_adjusted_close()
            self.compute_ndr_factor()
            self.normalize_factors()
            self.decay()
            self.industry_neutral()
            self.industry_even_neutralize()
            self.industry_size_neutral()
            self.calculate_adj_vwap()
            # self.optimal_portfolio()
            all_results=[]
            methods = ['normalized_factor', 'decayed_alpha', 'neutral_alpha','demean_industry_normalized','industry_size_neutral_alpha','winsorized_factor']
            # methods = ['demean_industry_normalized','industry_size_neutral_alpha','winsorized_factor']

            for method_name in methods:
                if method_name=='winsorized_factor':
                    self.mark_limit_and_suspension_conditions()
                    self.winsorized_factor()
                method_data = self.data[method_name]
                """
                simple backtest
                """
                pnl_df = self.simple_backtest(method_data)



                # Calculate parameter using the method data
                parameter_df,pnl= self.calculate_parameter(method_data)
                parameter_df=self.load_index_data(parameter_df)
                print(f"Profit/Loss from trading based on {method_name}:", pnl)

                # Save the parameter DataFrame to CSV
                parameter_df.to_csv(f'/Users/zw/Desktop/result_zyb_pnl_HS300{method_name}.csv', index=False)


                """
                compound interest backtest
                """

                # pnl_df, overall_pnl, long_pnl, short_pnl = self.compoundInterestBacktest(method_data)
                # print(method_data.name)
                # pnl_df = self.load_index_data(pnl_df)
                # print(f"Profit/Loss from trading based on {method_name}:")
                # print(f"  Overall P&L: {overall_pnl}")
                # print(f"  Long P&L: {long_pnl}")
                # print(f"  Short P&L: {short_pnl}")
                #
                # # self.plot_pnl(pnl_df, method_name)
                # pnl_df['method'] = method_name  # Add a column to identify the method
                # all_results.append(pnl_df)
                #
                # # self.plot_pnl(pnl_df, method_name)
                # pnl_df.to_csv(f'/Users/zw/Desktop/玄元投资实习/玄元TASK1/resultHS300_{method_name}.csv', index=False,
                #               encoding='utf-8-sig')

                """
                for debugging
                """
                self.data.sort_values(by='TRADINGDAY_x', inplace=True)
                result_df = self.data[
                    ['TRADINGDAY_x', 'SECU_CODE', 'next_day_return','VWAP','ADJ_VWAP','hold_pnl','trade_pnl','long_capital_allocation','short_capital_allocation','ADJ_CLOSE_PRICE']].copy()
                result_df.to_csv(f'/Users/zw/Desktop/debug_{method_name}.csv', index=False)
                print(f"Results saved to file.csv")
        else:
            print("No data loaded.")

if __name__ == "__main__":
    file_path_sample = "/Users/zw/Desktop/sample_data.csv"
    file_path_all="/Users/zw/Desktop/combined_data.csv"
    file_path_HS300 = "/Users/zw/Desktop/combinedHS300_data.csv"
    file_path_HS300_recent="/Users/zw/Desktop/sampleHS300_data.csv"
    analysis = NDR(file_path_HS300)
    analysis.run_analysis()
