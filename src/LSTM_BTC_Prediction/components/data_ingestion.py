import pandas as pd 
import os 
from LSTM_BTC_Prediction import logger 
import yfinance as yf 
from datetime import datetime
import talib
import requests
from lxml import html
import re
from LSTM_BTC_Prediction.entity.config_entity import DataIngestionConfig




class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_BTC(self) :
        try:
            symbol = 'BTC-USD'
            start_date = '2014-09-17'
            end_date = self.config.current_date
            data = yf.download(symbol, start=start_date, end=end_date)
            df = pd.DataFrame({
                'DateTime': data.index,
                'close': data['Close'],
                'volume': data['Volume'],
                'open': data['Open'],
                'high': data['High'],
                'low': data['Low']
            })

            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df.set_index(df['DateTime'], inplace=True)
            df.drop(columns='DateTime', inplace=True)
            self.df=df
            logger.info(f"Downloaded data for BTC-USD")


        except Exception as e:
            logger.error(f"Error occurred during data download for BTC-USD : {e}")
            raise e


    def adding_indicators(self):

        self.df['SMA'] = talib.SMA(self.df['close'], timeperiod=5)
        self.df['EMA'] = talib.EMA(self.df['close'], timeperiod=5)
        self.df['WMA'] = talib.WMA(self.df['close'], timeperiod=5)
        self.df['SMMA'] = talib.SMA(self.df['close'], timeperiod=5)
        self.df['target'] = self.df['close'] - self.df['open']
        self.df['target_class'] = [1 if self.df['target'][i] > 0 else 0 for i in range(len(self.df))]
        self.df['ATR'] = talib.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)
        self.df['OBV'] = talib.OBV(self.df['close'], self.df['volume'])
        self.df['daily_return'] = self.df['close'].pct_change()
        self.df['feb_0.236'] = self.df['low'] + 0.236 * (self.df['high'] - self.df['low'])
        self.df['feb_0.382'] = self.df['low'] + 0.382 * (self.df['high'] - self.df['low'])
        self.df['feb_0.5'] = self.df['low'] + 0.5 * (self.df['high'] - self.df['low'])
        self.df['feb_0.618'] = self.df['low'] + 0.618 * (self.df['high'] - self.df['low'])
        self.df['feb_1'] = self.df['low'] + (self.df['high'] - self.df['low'])

        logger.info("Added indicators to DataFrame")

 


    def download_Transaction(self) :
        try:
            response = requests.get(self.config.source_URL_Transaction)
            if response.status_code == 200:
                tree = html.fromstring(response.content)
                extracted_text_list = tree.xpath('//a/following-sibling::text()')
                extracted_text = ' '.join(map(str.strip, extracted_text_list)).replace(u'\xa0', ' ')
                pattern = re.compile(r'(\d{2}-\w{3}-\d{4} \d{2}:\d{2})\s+(-?\d+[KM]?)')
                matches = pattern.findall(extracted_text)
                df = pd.DataFrame(matches, columns=['DateTime', 'Transactions'])
                df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d-%b-%Y %H:%M')
                df['DateTime'] = df['DateTime'].dt.strftime('%Y-%m-%d')

                compare_date = pd.to_datetime("2014-10-01")
                df["DateTime"] = pd.to_datetime(df["DateTime"])
                df = df[df["DateTime"] >= compare_date]

                df.set_index(df['DateTime'], inplace=True)
                df.drop(columns='DateTime', inplace=True)

                df['Transactions'] = df['Transactions'].apply(
                    lambda x: int(float(x.replace('M', '')) * 1000000) if 'M' in x else (
                        int(float(x.replace('K', '')) * 1000) if 'K' in x else
                        int(float(x))
                    ))

                self.df= self.df[self.df.index >= compare_date]

                self.df = self.df.join(df)
            else:
                logger.error(f"Error occurred during data download for TRANSACTION in response: {response.status_code}")

            logger.info(f"Downloaded data for TRANSACTION")

        except Exception as e:
            logger.error(f"Error occurred during data download for TRANSACTION : {e}")
            raise e


    def download_Blocks(self) :
        try:
            response = requests.get(self.config.source_URL_blocks)

            if response.status_code == 200:
                tree = html.fromstring(response.content)
                extracted_text_list = tree.xpath('//a/following-sibling::text()')
                extracted_text = ' '.join(map(str.strip, extracted_text_list)).replace(u'\xa0', ' ')
                pattern = re.compile(r'(\d{2}-\w{3}-\d{4} \d{2}:\d{2})\s+(-?\d+[KM]?)')
                matches = pattern.findall(extracted_text)

                df = pd.DataFrame(matches, columns=['DateTime', 'Blocks'])
                df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d-%b-%Y %H:%M')
                df['DateTime'] = df['DateTime'].dt.strftime('%Y-%m-%d')

                compare_date = pd.to_datetime("2014-10-01")
                df["DateTime"] = pd.to_datetime(df["DateTime"])
                df = df[df["DateTime"] >= compare_date]

                df.set_index(df['DateTime'], inplace=True)
                df.drop(columns='DateTime', inplace=True)

                df['Blocks'] = df['Blocks'].apply(
                    lambda x: int(float(x.replace('K', '')) * 1000) if 'K' in x else int(float(x)))

                self.df = self.df.join(df)

            else:
                logger.error(f"Error occurred during data download for BLOCKS in response: {response.status_code}")

            logger.info(f"Downloaded data for BLOCKS")

        except Exception as e:
            logger.error(f"Error occurred during data download for Blocks : {e}")
            raise e


    def download_INT_RATE(self) :
        try:
            symbol = '^IRX'
            start_date = '2014-09-17'
            end_date = self.config.current_date
            interest_rates = yf.download(symbol, start='2014-09-17', end=end_date)
            df = pd.DataFrame({
                'DateTime': interest_rates.index,
                'INT_Rate': interest_rates['Close']
            })

            expected_dates = pd.date_range(start='2014-09-17', end=end_date, freq='D')
            df = df.reindex(expected_dates)
            df['INT_Rate'] = df['INT_Rate'].fillna(method='ffill')
            df = df.iloc[:-1]
            compare_date = pd.to_datetime("2014-10-01")
            df["DateTime"] = pd.to_datetime(df.index)
            df = df[df["DateTime"] >= compare_date]
            df.drop(columns='DateTime', inplace=True)
            self.df = self.df.join(df)

            logger.info(f"Downloaded data for INT_RATE")

        except Exception as e:
            logger.error(f"Error occurred during data download for INT_RATE : {e}")
            raise e


    def download_STOCK_PRICE(self) :
        try:
            symbol = "^GSPC"
            start_date = '2014-09-17'
            end_date = self.config.current_date
            stock_indices = yf.download(symbol, start=start_date, end=end_date)
            df = pd.DataFrame({
                'DateTime': stock_indices.index,
                'open_stk': stock_indices['Open'],
                'high_stk': stock_indices['High'],
                'low_stk': stock_indices['Low'],
                'close_stk': stock_indices['Close'],
                'volume_stk': stock_indices['Volume']
            })

            expected_dates = pd.date_range(start='2014-09-17', end=end_date, freq='D')
            df = df.reindex(expected_dates)
            df['open_stk'] = df['open_stk'].fillna(method='ffill')
            df['high_stk'] = df['high_stk'].fillna(method='ffill')
            df['low_stk'] = df['low_stk'].fillna(method='ffill')
            df['close_stk'] = df['close_stk'].fillna(method='ffill')
            df['volume_stk'] = df['volume_stk'].fillna(method='ffill')
            df = df.iloc[:-1]
            compare_date = pd.to_datetime("2014-10-01")
            df["DateTime"] = pd.to_datetime(df.index)
            df = df[df["DateTime"] >= compare_date]
            df.drop(columns='DateTime', inplace=True)
            self.df = self.df.join(df)

            logger.info(f"Downloaded data for STOCK_PRICE")

        except Exception as e:
            logger.error(f"Error occurred during data download for STOCK_PRICE : {e}")
            raise e


    def download_INFLATION(self) :
        try:
            symbol = 'TIP'
            start_date = '2014-09-17'
            end_date = self.config.current_date
            inflation_data = yf.download(symbol, start=start_date, end=end_date)
            df = pd.DataFrame({
                'DateTime':inflation_data.index,
                'inflation': inflation_data['Close']
            })
            expected_dates = pd.date_range(start='2014-09-17', end=end_date, freq='D')
            df = df.reindex(expected_dates)
            df['inflation'] = df['inflation'].fillna(method='ffill')
            df = df.iloc[:-1]

            compare_date = pd.to_datetime("2014-10-01")
            df["DateTime"] = pd.to_datetime(df.index)
            df = df[df["DateTime"] >= compare_date]
            df.drop(columns='DateTime', inplace=True)
            
            self.df= self.df.join(df)

            logger.info(f"Downloaded data for INFLATION")

        except Exception as e:
            logger.error(f"Error occurred during data download for INFLATION : {e}")
            raise e


    def save_dataset(self):
        os.makedirs("artifacts/data_ingestion", exist_ok=True)
        file_path = os.path.join(self.config.root_dir, f"{self.config.dataset_name}.csv")
        self.df.to_csv(file_path)



