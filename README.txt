A single Python program for downloading iNaturalist images, reviewing YOLO detections, and exporting labelled datasets.

-Install

pip install -r requirements.txt

-Run

Python scrape.py

Use the GUI to search for a species, choose the number of images, download/resume a session, review detections with your YOLO model, and export approved images to a standard YOLO dataset.

Review controls: left-drag - add box, right-click - delete, 1/2 - species/person, Enter - save, S - reject, R - reset detections, Q - quit.

Data is stored in a folder called inat_data. Existing folders can also be imported from the GUI.

Be careful with the species names as there are many sub-species, so specificity is needed. For example, searching 'fox' results a wide range of types of foxes. For my use case, a search of 'vulpes vulpes' or 'red fox' provides a more useful result.

Additionally, a YOLO model is needed for review. For species identified within the dissertation, that model can be used. For species outside of that, the standard COCO pre-trained YOLO model can be used, but if the species fall outside of the COCO classes also, it may be worth training an intermediate model on a smaller subset of data. Pretrained COCO YOLO can be used with confidence set low in similar classes in the meantime.