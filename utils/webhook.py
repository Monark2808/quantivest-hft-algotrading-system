import requests

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1385350319327805542/sXNvH1ZyCGujwYDgSwH5tuHjKJW9Uuq_v4tqMMRA5KEkVgnGeO_ppYXxlHFT6YBuvZEX"

def send_trade_alert(prediction, features, confidence_dict, webhook_url, symbol):
    content = (
        f"🚨 **Trade Alert: {prediction} — {symbol}**\n"
        f"**Features:**\n"
        f"- MAP: {features['map_prob']:.2f}\n"
        f"- OVII: {features['ovii']:.2f}\n"
        f"- IVDI: {features['ivdi']:.2f}\n"
        f"- Return: {features['return']:.5f}\n\n"
        f"**Model Confidence:**\n"
        + "\n".join([f"- {k}: {v:.1%}" for k, v in confidence_dict.items()])
    )

    payload = {"content": content}
    try:
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 204:
            print("✅ Discord alert sent successfully.")
        else:
            print(f"⚠️ Discord alert failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error sending Discord alert: {e}")
