.PHONY: clean, setup

ecseg:
	chmod +x *.sh && ./setup.sh
	
metaseg:
	python src/metaseg.py

meta_overlay:
	python src/meta_overlay.py

interseg:
	python src/interseg.py

clean : 
	rm -r __pycache__
