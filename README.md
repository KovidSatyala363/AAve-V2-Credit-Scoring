# Aave Credit Scoring

This project performs credit scoring for Ethereum wallets using their transaction history with the Aave V2 protocol. It reads transaction data in JSON format, extracts features such as transaction frequency and volume, and computes a normalized credit score for each wallet.

## Features

- Loads DeFi transaction data from a JSON file
- Extracts wallet-level features:
  - Count of actions (deposit, withdraw, borrow, repay)
  - Total and average amounts per action
- Applies normalization using MinMaxScaler
- Outputs results to a CSV file

## Project Structure

aave-credit-scoring/
│
├── data/
│ └── user_transactions.json # Input transaction data
│
├── outputs/
│ └── wallet_scores.csv # Output scores (generated after running)
│
├── src/
│ └── scoring.py # Main processing and scoring script
│
├── .gitignore
├── requirements.txt
└── README.md

To run : python src/scoring.py
![image alt](https://github.com/KovidSatyala363/AAve-V2-Credit-Scoring/blob/7b4f1d9863efb883dc680f629093d20f2d696c0e/Screenshot%202025-07-17%20093152.png)


