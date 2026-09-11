DOMAIN_MAP = {
    "HDFCBANK": "hdfcbank.com", "ICICIBANK": "icicibank.com", 
    "TCS": "tcs.com", "INFY": "infosys.com",
    "MARUTI": "marutisuzuki.com", "M&M": "mahindra.com",
    "ITC": "itcportal.com", "HINDUNILVR": "hul.co.in",
    "SUNPHARMA": "sunpharma.com", "DRREDDY": "drreddys.com",
    "RELIANCE": "ril.com", "NTPC": "ntpc.co.in",
    "TATASTEEL": "tatasteel.com", "HINDALCO": "hindalco.com",
    "HAL": "hal-india.co.in", "BEL": "bel-india.in", 
    "MAZDOCK": "mazagondock.in", "RVNL": "rvnl.org",
    "IRFC": "irfc.nic.in", "IREDA": "ireda.in",
    "SUZLON": "suzlon.com", "ETERNAL": "zomato.com",
    "YESBANK": "yesbank.in", "IDEA": "myvi.in",
    "GOLDBEES": "nipponindiaim.com", "SILVERBEES": "nipponindiaim.com",
    "NIFTYBEES": "nipponindiaim.com",
    "TATAGOLD": "tata.com", "TATSILV": "tata.com",
    "HDFCMID150": "hdfcfund.com"
}

def get_domain(ticker):
    return DOMAIN_MAP.get(ticker, "nseindia.com")
