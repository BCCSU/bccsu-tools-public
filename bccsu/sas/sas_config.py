# The java I need
# https://www.azul.com/downloads/?version=java-8-lts&os=windows&architecture=x86-64-bit&package=jdk#zulu
import os, saspy

SAS_config_names = ["wsl_ssh"]

wsl_ssh = {
    'ssh': r'C:\Windows\System32\OpenSSH\ssh.exe',  # or your preferred ssh client
    'host': '10.10.10.56',  # or WSL2 IP if needed
    'port': 22,
    # 'tunnel': 11912, # Important not to set. It sets hostname to localhost for some reason.
    'rtunnel': 11913,
    'luser': 'cgrant',
    'identity': r'C:\Users\cgrant\.ssh\id_ed25519',  # uncomment if using key auth
    'saspath': '/usr/local/SASHome/SASFoundation/9.4/sas',
    'options': ['-fullstimer'],
    'encoding': 'utf-8',
    'tdir': '/tmp',
    'localhost': '10.10.10.53',
    # 'wait': 60
}
