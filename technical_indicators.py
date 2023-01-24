# Trading w/ ML - Springboard - Capstone 3 
import pandas as pd
import numpy as np


pd.set_option('display.max_rows', 100)

"""
Things to consider when making technicals for a ML algorithm

1) We want these to be independent of time, so things like % change, or 
a moving average ratio with the price are fine, while indicators that
will shrink or grow with the price are bad.

2) We want to have calculation_window be an input to the function,
where applicable

3) Stationarity

"""

def calc_returns(price_ser: pd.Series, window: int=1):
	"""
	Calculate window period returns for a price series
	"""
	return_ser = (price_ser / price_ser.shift(periods=window)) - 1
	return return_ser


def calc_log_return_series(price_ser: pd.Series, window: int=1):
	"""
	Calculate window period log returns for a price series
	"""
	log_return_ser = np.log(price_ser / price_ser.shift(periods=window))

	return log_return_ser


def calculate_volatility(price_ser: pd.Series, window: int):
	"""
	Calculate the rolling standard deviation over a specified window
	"""
	return_ser = calc_returns(price_ser=price_ser, window=1)
	vol_ser = return_ser.rolling(window=window).std()

	return vol_ser


def calculate_rsi(price_ser: pd.Series, window: int):
	"""
	Calculate the rolling relative strength index using exponentially weighted average 
	alpha = 1/3 should be reviewed; must pass com, span, halflife or alpha into .ewm() method
	"""
	price_delta = price_ser.diff(1)
	
	gain_ser = price_delta.clip(lower=0)
	loss_ser = price_delta.clip(upper=0)	

	#avg_gain_ser = gain_ser.rolling(window).mean()
	#avg_loss_ser = loss_ser.rolling(window).mean()
	avg_gain_ser = gain_ser.ewm(alpha=1/3, min_periods=window).mean()
	avg_loss_ser = loss_ser.ewm(alpha=1/3, min_periods=window).mean()

	rs_ser = avg_gain_ser / abs(avg_loss_ser)
	rsi_ser = 100 - (100 / (1 + rs_ser))

	return price_delta, gain_ser, loss_ser, avg_gain_ser, avg_loss_ser, rs_ser, rsi_ser


def calculate_rsi_w_smoothing_avg(price_ser: pd.Series, window: int):
	"""
	Calculate the relative strength index of last n periods with smoothing average
	"""
	df = pd.DataFrame(price_ser)

	df['diff'] = df.diff(1)

	df['gain'] = df['diff'].clip(lower=0)
	df['loss'] = df['diff'].clip(upper=0).abs()
	
	
	for i, row in enumerate(range(len(price_ser))):
		if i <= window:
			df['avg_gain'] = df['gain'].rolling(window=window).mean()
			df['avg_loss'] = df['loss'].rolling(window=window).mean()
		else: 
			df['avg_gain'].iloc[i] = ((df['avg_gain'].iloc[i-1] * (window-1)) + df['gain'].iloc[i]) / window
			df['avg_loss'].iloc[i] = (df['avg_loss'].iloc[i-1] * (window-1) + df['loss'].iloc[i]) / window



	df['rs'] = df['avg_gain'] / df['avg_loss']
	df['rsi'] = 100 - (100 / (1.0 + df['rs']))

	return df


# Calculate annualized volatility

def get_years_past(series: pd.Series) -> float:
    """
    Calculate the years past according to the index of the series for functions that require annualization
    """
    
    start_date = series.index[0]
    end_date = series.index[-1]
    
    # Note: Had to convert the TimeDelta by pd.to_timedelta(1, unit='D') in order to obtain an integer for years_past
    return ((end_date - start_date) / 365.25) / pd.to_timedelta(1, unit='D')

def calc_annualized_volatility(return_series: pd.Series) -> float:
    """
    Calculates annualized volatility for a date-indexed return series.
    Works for any interval of date-indexed prices and returns.
    """
    
    years_past = get_years_past(return_series)
    entries_per_year = return_series.shape[0] / years_past
    
    return return_series.std() * np.sqrt(entries_per_year)


def calc_sma(series: pd.Series, n: int=20) -> pd.Series:
    """
    Calculates the simple moving average with pandas
    """
    
    return series.rolling(n).mean()


def calc_macd(series: pd.Series, n1: int=5, n2: int=34) -> pd.Series:
    """
    Calculate the MACD oscillator, given a short moving avg of length n1 and long moving avg of length n2
    """
    assert n1 < n2, f'n1 must be less than n2'
    
    return calc_sma(series, n1) - calc_sma(series, n2)


def calc_bollinger_bands(series: pd.Series, n: int=20) -> pd.DataFrame:
    """
    Calculates the bollinger bands and returns them as a dataframe
    """
    
    sma = calc_sma(series, n)
    std = calc_smstd(series, n)
    
    return pd.DataFrame({'middle': sma, 'upper': sma + 2 * std, 'lower': sma - 2 *std})


def PROC(ser: pd.Series, n: int=9):
    """
    calculates the indicator price rate of change of the series
    """
    
    return ser.pct_change(periods=n)


def r_percent(ser:pd.Series, n: int=14):
    """
    calculates the williams r_percent indicator
    """
    
    return 100 * ((roll_high(ser) - ser) / (roll_high(ser) - roll_low(ser)))


def k_percent(ser: pd.Series, n: int=14):
    """
    calculates the stochastic oscillator or k_percent indicator
    """
    
    return 100 * ((ser - roll_low(ser)) / (roll_high(ser) - roll_low(ser)))


def roll_high(ser: pd.Series, n: int=14):
    """
    find the rolling max of the last n days
    """
    
    return ser.rolling(n).max()


def roll_low(ser: pd.Series, n: int=14):
    """
    find the rolling max of the last n days
    """
    
    return ser.rolling(n).min()


def calc_bollinger_bands(series: pd.Series, n: int=20) -> pd.DataFrame:
    """
    Calculates the bollinger bands and returns them as a dataframe
    """
    
    sma = calc_sma(series, n)
    std = calc_smstd(series, n)
    
    return pd.DataFrame({'middle': sma, 'upper': sma + 2 * std, 'lower': sma - 2 *std})


def calc_smstd(series: pd.Series, n: int=20) -> pd.Series:
    """
    Calculates the simple moving standard deviation of a series
    """
    return series.rolling(n).std()


def calc_macd(series: pd.Series, n1: int=5, n2: int=34) -> pd.Series:
    """
    Calculate the MACD oscillator, given a short moving avg of length n1 and long moving avg of length n2
    """
    assert n1 < n2, f'n1 must be less than n2'
    return calc_sma(series, n1) - calc_sma(series, n2)


def calc_sma(series: pd.Series, n: int=20) -> pd.Series:
    """
    Calculates the simple moving average with pandas
    """
    return series.rolling(n).mean()


# calculate compounded annual growth rate

def calc_cagr(series: pd.Series) -> float:
    """
    Calculate compunded annual growth rate
    """
    value_factor = series.iloc[-1] / series.iloc[0]
    #value_factor = float(value_factor)
    years_past = get_years_past(series)
    
    return (value_factor ** (1 / years_past)) - 1


def calc_sortino_ratio(price_series: pd.Series, benchmark_rate: float=0) -> float:
    """
    Calculates the sortino ratio
    """
    
    cagr = calc_cagr(price_series)
    return_series = calc_returns(price_series)
    downside_deviation = calc_annualized_downside_deviation(return_series)
    
    return (cagr - benchmark_rate) / downside_deviation


def calc_annualized_downside_deviation(return_series: pd.Series, benchmark_rate: float=0) -> float:
    """
    Calculates the downside deviation for use in the Sortino Ratio.
    
    Benchmark rate is assumed to be annualized. It will be adjusted according to the number of periods per year seen in the data.
    """
    
    # For both de-annualizing the benchmark rate and annualizing result
    years_past = get_years_past(return_series)
    entries_per_year = return_series.shape[0] / years_past
    
    adjusted_benchmark_rate = ((1 + benchmark_rate) ** (1/entries_per_year)) - 1
    
    downside_series = adjusted_benchmark_rate - return_series
    downside_sum_of_squares = (downside_series[downside_series > 0] ** 2).sum()
    denominator = return_series.shape[0] - 1
    downside_deviation = np.sqrt(downside_sum_of_squares / denominator)
    
    return downside_deviation * np.sqrt(entries_per_year)


def calc_sharpe_ratio(price_series: pd.Series, benchmark_rate: float=0) -> float:
    """
    Calculates the sharpe ratio given a price series. Defaults to benchmark_rate of zero
    """
    cagr = calc_cagr(price_series)
    return_series = calc_returns(price_series)
    volatility = calc_annualized_volatility(return_series)
    return (cagr - benchmark_rate) / volatility


def OBV(ser: pd.Series, vol_ser: pd.Series):
    """
    calculate the on balance volume of the close price for the dataframe
    """
    
    return pd.Series(np.where(ser > ser.shift(1), vol_ser, np.where(ser < ser.shift(1), -vol_ser, 0)).cumsum(), index=df_AWU.index)


