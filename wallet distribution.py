# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:56:11 2025

@author: Admin
"""
import json
import base58
from solders.keypair import Keypair

def create_wallet(n=10):
    wallets = {}
    for i in range(n):
        keypair = Keypair()
        address = str(keypair.pubkey())  # Use .pubkey() instead of .public_key
        private_key = base58.b58encode(bytes(keypair)).decode("utf-8")  # Convert to bytes before encoding
        wallets[i]={'address': address, 'private_key': private_key}
    return wallets

wallets = create_wallet(10)

# %%
from solana.rpc.api import Client
from solana.transaction import Transaction
from solana.system_program import TransferParams, transfer

client = Client("https://api.mainnet-beta.solana.com")  # Or use testnet/devnet for testing

# Get the first address to send transactions from
sender_private_key = wallets[0]["private_key"]
sender_keypair = Keypair.from_secret_key(bytes.fromhex(sender_private_key))

# %%
# Prepare transaction data
def send_bulk_transactions(sender_keypair, address_dict, amount):
    # Iterate over all addresses (except the sender address)
    for addr, details in address_dict.items():
        if addr == "address_1":  # Skip the sender address
            continue
        
        recipient_public_key = details["public_key"]
        recipient_keypair = Keypair.from_public_key(bytes.fromhex(recipient_public_key))

        # Create transaction
        transaction = Transaction()
        transaction.add(
            transfer(TransferParams(
                from_pubkey=sender_keypair.public_key,
                to_pubkey=recipient_keypair.public_key,
                lamports=amount  # The amount to send in lamports (1 SOL = 1 billion lamports)
            ))
        )
        
        # Send the transaction
        response = client.send_transaction(transaction, sender_keypair)
        print(f"Transaction sent to {recipient_public_key}: {response}")

# Example: Send 1 SOL (1 SOL = 1 billion lamports)
send_bulk_transactions(sender_keypair, address_dict, 1_000_000_000)  # 1 SOL in lamports