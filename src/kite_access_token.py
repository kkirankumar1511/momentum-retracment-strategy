from kiteconnect import KiteConnect
import requests
import hashlib
from config import load_config

def get_access_token():
    cfg = load_config()
    api_key = cfg['kite']['api_key']
    request_token = cfg['kite']['request_token']
    api_secret = cfg['kite']['api_secret']
    api_url = cfg['kite']['api_url']
    # 1. Create the checksum
    data_to_hash = api_key + request_token + api_secret
    # Use encode() to convert the string to bytes, which is required by hashlib
    checksum = hashlib.sha256(data_to_hash.encode("utf-8")).hexdigest()
    # 2. Prepare the POST request data
    payload = {
        "api_key": api_key,
        "request_token": request_token,
        "checksum": checksum
    }
    # 3. Make the POST request
    response = requests.post(api_url, data=payload)
    # 4. Process the response
    if response.status_code == 200:
        session_data = response.json().get("data", {})
        access_token = session_data.get("access_token")
        print(f"Successfully generated session!")
        print(f"Access Token: {access_token}")
        # Now you can initialize the KiteTicker or KiteConnect object
        # from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        print(kite.profile())
    else:
        print(f"Authentication failed!")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")


if __name__ == "__main__":
    get_access_token()