# Define the default target
default: run

# Define a target to run the Python script
run:
	. /projects/ec232/venvs/in5310/bin/activate && python validate_project1.py

