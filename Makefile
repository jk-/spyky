.PHONY: init

init:
	pip install -r requirements.txt

clean:
	find . -name '*.pyc' -delete

tests: clean
	pytest -p no:warnings -vv
