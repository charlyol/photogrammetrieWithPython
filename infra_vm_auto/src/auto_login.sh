#!/usr/bin/expect -f

set timeout -1

# Variables
set username "regnilo"
set ip_address "192.168.122.42"
set ssh_key_path "/home/olinger/.ssh/regnilo_key"

# Start SSH connection
spawn ssh -i $ssh_key_path $username@$ip_address

# Handle the SSH key acceptance
expect {
    "yes/no" { send "yes\r"; exp_continue }
    "password:" { exit 1 }
}

# Interact with the user after login
interact
