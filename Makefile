.PHONY: clean, setup

ecseg:
	chmod +x *.sh && ./setup.sh
	
metaseg:
	python src/metaseg.py

meta_overlay:
	python src/meta_overlay.py

nuclei_fish:
	python src/nuclei_fish.py

fish_distance_calculation:
	python src/fish_distance_calculation.py

interseg:
	python src/interseg.py

clean : 
	rm -r __pycache__
