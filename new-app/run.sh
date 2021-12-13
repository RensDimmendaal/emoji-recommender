#!/usr/bin/env bash
docker container kill emoji
docker container rm emoji
docker container run -d --name emoji -p 80:80 emoji
docker container logs emoji --follow