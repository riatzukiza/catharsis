
FROM node:14
FROM tensorflow/tensorflow:latest-gpu

ADD ./setup.sh /setup.sh
RUN bash /setup.sh
WORKDIR /app


# RUN export NVM_DIR="$HOME/.nvm"
# RUN [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
# RUN [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# RUN nvm install 14
# RUN nvm use 14

# RUN npm install
# RUN npm install -g typescript ts-node

