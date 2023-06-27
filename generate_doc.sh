#!/bin/bash
rm -Rf docs/html/*
sphinx-apidoc -f -o docs/ validating_models --separate
sphinx-build docs/ docs/html/
