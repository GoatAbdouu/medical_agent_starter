.PHONY: help install run test clean

help:
	@echo "Commandes disponibles:"
	@echo "  make install - Installer"
	@echo "  make run     - Lancer"
	@echo "  make test    - Tester"
	@echo "  make clean   - Nettoyer"

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

test:
	python3 scripts/test_system.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
