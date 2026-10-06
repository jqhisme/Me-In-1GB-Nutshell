# Setup Guide for MeIn512mbNutshell

## 1 Install Armbian
Download [Armbian Imager](https://imager.armbian.com/#downloads) and select an [image](https://armbian.com/boards/radxa-zero) to burn.

When burning, write the wifi and password for ssh connection.

## 2 SSH into the board
Use `arp -a` to find the ip address for the board. Then SSH into the board using 
```
ssh radxa@<ip-address>
```
The initial password is `1234`

After ssh into the board, update everything by running 
```
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

## 3 Verify HID support
Use an AI agent ot do this

## 4 Install llama.cpp
First install cmake
```bash
sudo apt install -y git cmake build-essential python3
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-cli llama-server llama-completion -j 1
./build/bin/llama-cli --help
```

## 5 Clone code from github repo

## 6 Test the inference

## 7 Setup the screen

## 8 Setup auto start with systemctl
