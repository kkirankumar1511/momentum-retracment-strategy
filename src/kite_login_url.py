from kiteconnect import KiteConnect
from config import load_config


def get_login_url():
    cfg = load_config()
    api_key = cfg['kite']['api_key']
    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()
    print(f"Please visit this URL to login: {login_url}")


if __name__ == "__main__":
    get_login_url()