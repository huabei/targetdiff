#!/usr/bin/bash
echo -n wanghuabei123 | xclip -sel clip
rsync -av /home/huabei/project/github/targetdiff wanghuabei@192.168.0.172:/home/wanghuabei/project --exclude=logs* --exclude=log* --exclude=ckpt --exclude=outputs* --exclude=.env --exclude=*.tgz --exclude=*.tar.gz --exclude=local --exclude=pdbbind --exclude=*.pdb --exclude=*.sdf --exclude=notebooks
