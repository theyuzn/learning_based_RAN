FROM ubuntu:jammy-20230308
LABEL yuzn star127532@gmail.com

RUN apt update
RUN apt upgrade -y
RUN apt install build-essential git -y
RUN apt install python3 -y
RUN apt install python3-pip -y
RUN apt libsctp-dev -y
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install scipy numpy gym matplotlib
RUN apt install net-tools

