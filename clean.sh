#!/bin/bash

# Define the directory
DIRECTORY="plot/"

# Check if the directory exists
if [ -d "$DIRECTORY" ]; then
    # Remove all files in the directory
    rm -f ${DIRECTORY}*
    echo "All files in ${DIRECTORY} have been deleted."
else
    echo "Directory ${DIRECTORY} does not exist."
fi
