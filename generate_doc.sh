#!/bin/bash
rm -R docs/html/*
sphinx-apidoc -f -o sphinx/ validating_models --separate
sphinx-build sphinx/ docs/html/