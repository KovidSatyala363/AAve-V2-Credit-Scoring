import json
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def load_transactions(file_path):
    with open(file_path, 'r') as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        data = [raw_data]
    else:
        data = raw_data

    records = []
    for tx in data:
        record = {}

        wallet = tx.get('wallet') or tx.get('user') or tx.get('address')
        if not wallet and 'user' in tx.get('params', {}):
            wallet = tx['params'].get('user')
        record['wallet'] = wallet if wallet else 'unknown'

        record['action'] = tx.get('action') or tx.get('type') or tx.get('method') or 'unknown'

        amount = 0
        if isinstance(tx.get('params'), dict):
            amount = tx['params'].get('amount') or tx['params'].get('value') or 0
        amount = amount or tx.get('amount') or 0
        try:
            record['amount'] = float(amount)
        except:
            record['amount'] = 0.0

        record['timestamp'] = tx.get('timestamp') or tx.get('time') or None

        records.append(record)

    return pd.DataFrame(records)

def build_wallet_features(df):
    df['amount'] = df['amount'].fillna(0).astype(float)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')

    grouped = df.groupby('wallet')

    features = pd.DataFrame()
    features['total_txns'] = grouped.size()
    features['total_amount'] = grouped['amount'].sum()
    features['average_amount'] = grouped['amount'].mean()
    features['max_amount'] = grouped['amount'].max()
    features['min_amount'] = grouped['amount'].min()

    actions = ['deposit', 'borrow', 'repay', 'redeemunderlying', 'liquidationcall']
    for action in actions:
        action_counts = df[df['action'] == action].groupby('wallet').size()
        features[f'{action}_count'] = features.index.map(action_counts).fillna(0).astype(int)

    features['borrow_to_repay_ratio'] = features['borrow_count'] / (features['repay_count'] + 1)
    features['liquidation_ratio'] = features['liquidationcall_count'] / (features['total_txns'] + 1)

    return features.fillna(0).reset_index()

def calculate_scores(features_df):
    safe_features = [
        'total_txns', 'total_amount', 'average_amount',
        'max_amount', 'min_amount',
        'deposit_count', 'repay_count', 'redeemunderlying_count'
    ]

    risk_features = [
        'borrow_to_repay_ratio', 'liquidation_ratio',
        'borrow_count', 'liquidationcall_count'
    ]

    scaler = MinMaxScaler()
    features_df[safe_features] = scaler.fit_transform(features_df[safe_features])

    features_df['safe_score'] = features_df[safe_features].mean(axis=1)
    features_df['risk_score'] = 1 - features_df[risk_features].mean(axis=1)

    features_df['credit_score'] = (
        0.7 * features_df['safe_score'] +
        0.3 * features_df['risk_score']
    ) * 1000

    features_df['credit_score'] = features_df['credit_score'].clip(0, 1000)

    return features_df[['wallet', 'credit_score']]

def main():
    input_path = 'data/user_transactions.json'
    output_path = 'outputs/wallet_scores.csv'

    print("Loading transaction data...")
    df = load_transactions(input_path)

    print("Generating wallet-level features...")
    wallet_features = build_wallet_features(df)

    print("Calculating credit scores...")
    scored_wallets = calculate_scores(wallet_features)

    os.makedirs('outputs', exist_ok=True)
    scored_wallets.to_csv(output_path, index=False)
    print(f"Credit scores have been saved to '{output_path}'.")

if __name__ == "__main__":
    main()
