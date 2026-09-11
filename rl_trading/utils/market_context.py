from datetime import datetime, timedelta

def get_expiry_context():
    """
    Returns context about F&O expiries in the Indian Market.
    Tuesday: FinNifty
    Wednesday: Midcap Nifty
    Thursday: Nifty 50 / Bank Nifty
    """
    now = datetime.now()
    weekday = now.weekday() # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
    
    context = {
        "is_expiry_day": False,
        "expiry_type": None,
        "market_sentiment": "NORMAL"
    }
    
    if weekday == 1:
        context["is_expiry_day"] = True
        context["expiry_type"] = "FINNIFTY"
    elif weekday == 2:
        context["is_expiry_day"] = True
        context["expiry_type"] = "MIDCAP"
    elif weekday == 3:
        context["is_expiry_day"] = True
        context["expiry_type"] = "NIFTY_MAIN"
        
    return context

# 2026 NSE Holidays Implementation
NSE_HOLIDAYS_2026 = [
    "2026-01-26", # Republic Day
    "2026-03-03", # Mahashivratri
    "2026-03-20", # Holi
    "2026-04-03", # Good Friday
    "2026-04-14", # Dr. Baba Saheb Ambedkar Jayanti
    "2026-05-01", # Maharashtra Day
    "2026-08-15", # Independence Day
    "2026-09-17", # Ganesh Chaturthi
    "2026-10-02", # Mahatma Gandhi Jayanti
    "2026-10-18", # Dussehra
    "2026-11-08", # Diwali (Laxmi Pujan)
    "2026-12-25", # Christmas
]

def is_market_currently_open() -> bool:
    """
    Checks if the current time matches active NSE trading hours (09:15 to 15:30 IST)
    and validates against weekends and public holidays.
    """
    import pytz
    import os
    if os.getenv("IGNORE_MARKET_HOURS", "false").lower() == "true": 
        return True
        
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(tz)
    
    # 1. Weekend Check (0=Mon, ... 5=Sat, 6=Sun)
    if now.weekday() >= 5: 
        return False
        
    # 2. Holiday Check
    date_str = now.strftime("%Y-%m-%d")
    if date_str in NSE_HOLIDAYS_2026:
        return False
        
    # 3. Market Hours Check (09:15 AM to 03:30 PM)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_open <= now <= market_close
