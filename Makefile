.PHONY: test install run

test:
	pytest tests/ -v

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload