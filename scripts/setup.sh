#!/bin/sh
set -e

info()
{
    local green="\033[1;32m"
    local normal="\033[0m"
    echo "[${green}reprox-setup${normal}] $1"
}

install_package()
{
    if [ -x "$(command -v apk)" ];       then sudo apk add --no-cache "$1"
    elif [ -x "$(command -v apt-get)" ]; then sudo apt install "$1"
    elif [ -x "$(command -v dnf)" ];     then sudo dnf install "$1"
    elif [ -x "$(command -v zypper)" ];  then sudo zypper install "$1"
    elif [ -x "$(command -v pacman)" ];  then sudo pacman -Syu "$1"
    elif [ -x "$(command -v brew)" ];    then sudo brew install "$1"
    else echo "FAILED TO INSTALL PACKAGE: Package manager not found. You must manually install: $1">&2; fi
}

install_java8_package()
{
    if [ -x "$(command -v apk)" ];       then sudo apk add --no-cache openjdk8
    elif [ -x "$(command -v apt-get)" ]; then sudo apt install openjdk-8-jdk
    elif [ -x "$(command -v dnf)" ];     then sudo dnf install java-1.8.0-openjdk
    elif [ -x "$(command -v zypper)" ];  then sudo zypper in java-1_8_0-openjdk
    elif [ -x "$(command -v pacman)" ];  then sudo pacman -Syu java8-openjdk
    elif [ -x "$(command -v brew)" ];    then sudo brew install openjdk@8
    else echo "FAILED TO INSTALL PACKAGE: Package manager not found. You must manually install Java 8">&2; fi
}

get_tool()
{
    if ! type "$1" > /dev/null; then
        info "Could not find $1. Installing $1..."
        install_package "$1"
    else
        info "$1 found."
    fi
}

info "Starting GPT-REPROX testing environment setup..."

# Download dependencies for GPT-REPROX and MineDojo
## Git checks and setup
get_tool git
get_tool wget

## Anaconda/Miniconda checks and setup
if ! type conda > /dev/null; then
    info "Could not find Anaconda/Miniconda installation. Installing Anaconda..."
    _conda_url=https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
    curl $_conda_url | sh
else
    info "Anaconda/Miniconda installation found."
fi

## Java checks and setup
if type java > /dev/null; then
    info "Found a Java installation."
    java_exec=java
elif [[ -n "$JAVA_HOME" ]] && [[ -x "$JAVA_HOME/bin/java" ]];  then
    info "Found a Java installation through JAVA_HOME env variable." 
    java_exec="$JAVA_HOME/bin/java"
else
    info "No Java installation found."
fi

if [[ "$java_exec" ]]; then
    # Version check, with numbers together (e.g. 1.8.0 is 18)
    required_java_major_version=18
    java_major_version=$(java -version 2>&1 | head -1 | cut -d'"' -f2 | sed '/^1\./s///' | cut -d'.' -f1)
    valid_java_version=0; [[ $java_major_version -eq required_java_major_version ]] && valid_java_version=1

    if [[ "$valid_java_version" -eq 1 ]]; then
        info "Java version is in 1.8 range."
    else         
        info "Java version is not in 1.8 range."
    fi
fi

if ! [[ "$java_exec" ]] || [[ "$valid_java_version" -eq 0 ]]; then
    info "Installing Java..."
    install_java8_package

    if [[ -d /usr/lib/jvm/java-8-openjdk-amd64 ]]; then
        export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
        info "Java 8 installation is now complete!"
        java_home_set=1
    else
        info "Java 8 installation is now complete, check if JAVA_HOME is set to Java 8 JDK!"
        java_home_set=0
    fi
fi

## Headless execution setup
apt install x11-utils xvfb

if [[ -z "$DISPLAY" ]]; then
    info "Setting up headless execution display..."
    export DISPLAY=:99
    Xvfb $DISPLAY -screen 0 1024x768x24 > /dev/null 2>&1 &
fi

# Clone GPT-REPROX
if [[ -d GPT-REPROX ]]; then
    info "GPT-REPROX already exists. Skipping clone..."
else
    info "Starting fetch of GPT-REPROX..."
    git clone --recurse-submodules --progress --verbose https://github.com/claucambra/GPT-REPROX.git
fi

expected_env_name="minigpt4"

info "Setting up GPT-REPROX conda environment..."
if conda info --envs | grep -qw $expected_env_name; then
    info "GPT-REPROX conda environment already exists. Skipping creation..."
else
    info "Creating GPT-REPROX conda environment..."
    conda env create -f GPT-REPROX/MiniGPT4/environment.yml
fi

info "Activating GPT-REPROX conda environment..."
conda activate $expected_env_name

info "Fixing package versions required for MineDojo in environment..."
pip install setuptools==65.5.0 wheel==0.38.4

info "Installing GPT-REPROX pip deps..."
pip install stable_baselines3 gymnasium

info "Fetching pretrained Vicuna 7B materials..."
wget https://huggingface.co/wangrongsheng/MiniGPT4-7B/resolve/main/prerained_minigpt4_7b.pth

info "Setting weights in MiniGPT config"
sed -i 's:/path/to/vicuna/weights/:wangrongsheng/MiniGPT-4-LLaMA:g' GPT-REPROX/MiniGPT4/minigpt4/configs/models/minigpt4.yaml
sed -i "s:/path/to/pretrained/ckpt/:$PWD/pretrained_minigpt4.pth:g" GPT-REPROX/MiniGPT4/eval_configs/minigpt4_eval.yaml

info "Setting up MineDojo..."
git clone https://github.com/claucambra/MineDojo.git
pip install -e MineDojo/

info "Cleaning up MineDojo directory..."
rm -Rf MineDojo

info ""
info "Setup complete!"
info "You can run the GPT-REPROX test by executing:"
info "MINEDOJO_HEADLESS=1 python GPT-REPROX/test.py --cfg-path GPT-REPROX/MiniGPT4/eval_configs/minigpt4_eval.yaml"
info "If you want to run the test with a GUI, you can do so by executing the prior command without MINEDOJO_HEADLESS=1"

if [[ "$java_home_set" -eq 0 ]]; then
    info "JAVA_HOME is not set to Java 8 JDK!"
    info "Make sure JAVA_HOME is set to Java 8 JDK path as otheriwse MineDojo is unlikely to work correctly."
fi

info ""