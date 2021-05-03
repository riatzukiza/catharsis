curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

nvm install 14
nvm use 14

npm install
npm install -g typescript ts-node


mkdir app

apt update && apt upgrade
apt install -y git emacs

# install spacemacs
git clone https://github.com/syl20bnr/spacemacs ~/.emacs.d
