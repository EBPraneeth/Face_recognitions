
import os
import pickle
import time
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from face_recognitions.entity.config_entity import Train_Representation_Model_Config
from deepface.modules import representation, detection, modeling
from deepface.models.FacialRecognition import FacialRecognition

from face_recognitions import logger

class Train_Representation_Model :
    def __init__(self,config:Train_Representation_Model_Config) :
        self.config=config

    def list_of_image_path(self,path:Path):

        """
        List images in a given path
        Args:
            path (str): path's location
        Returns:
            images (list): list of exact image paths
        """
        images = []
        for r, _, f in os.walk(path):
            for file in f:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    exact_path = os.path.join(r, file)
                    images.append(exact_path)
        return images[0:15]


    def find_bulk_embeddings(
        self,
        employees: List[Path],
        model_name: str = "VGG-Face",
        target_size: tuple = (224, 224),
        detector_backend: str = "opencv",
        enforce_detection: bool = True,
        align: bool = True,
        expand_percentage: int = 0,
        normalization: str = "base",
        silent: bool = False,
    ):
        """
        Find embeddings of a list of images

        Args:
            employees (list): list of exact image paths

            model_name (str): facial recognition model name

            target_size (tuple): expected input shape of facial recognition model

            detector_backend (str): face detector model name

            enforce_detection (bool): set this to False if you
                want to proceed when you cannot detect any face

            align (bool): enable or disable alignment of image
                before feeding to facial recognition model

            expand_percentage (int): expand detected facial area with a
                percentage (default is 0).

            normalization (bool): normalization technique

            silent (bool): enable or disable informative logging
        Returns:
            representations (list): pivot list of embeddings with
                image name and detected face area's coordinates
        """
        representations = []
        for employee in tqdm(
            employees,
            desc="Finding representations",
            disable=silent,
        ):
            try:
                img_objs = detection.extract_faces(
                    img_path=employee,
                    target_size=target_size,
                    detector_backend=detector_backend,
                    grayscale=False,
                    enforce_detection=enforce_detection,
                    align=align,
                    expand_percentage=expand_percentage,
                )
            except ValueError as err:
                logger.error(
                    f"Exception while extracting faces from {employee}: {str(err)}"
                )
                img_objs = []

            if len(img_objs) == 0:
                logger.warn(f"No face detected in {employee}. It will be skipped in detection.")
                representations.append((employee, None, 0, 0, 0, 0))
            else:
                for img_obj in img_objs:
                    img_content = img_obj["face"]
                    img_region = img_obj["facial_area"]
                    embedding_obj = representation.represent(
                        img_path=img_content,
                        model_name=model_name,
                        enforce_detection=enforce_detection,
                        detector_backend="skip",
                        align=align,
                        normalization=normalization,
                    )

                    img_representation = embedding_obj[0]["embedding"]
                    representations.append((
                        employee,
                        img_representation,
                        img_region["x"],
                        img_region["y"],
                        img_region["w"],
                        img_region["h"]
                        ))

        return representations


    def representations(self):
       
        tic = time.time() 

        # -------------------------------
        if os.path.isdir(self.config.train_faces_dir) is not True:
            raise ValueError("Passed train_faces_dir does not exist!")

        model: FacialRecognition = modeling.build_model(self.config.model_name)
        target_size = model.input_shape
        # ---------------------------------------

        representations = []

        df_cols = [
        "identity",
        f"{self.config.model_name}_representation",
        "target_x",
        "target_y",
        "target_w",
        "target_h",
        ]

        # Ensure the proper pickle file exists
        if not os.path.exists(self.config.represent_file):
            with open(self.config.represent_file, "wb") as f:
                pickle.dump([], f)

        # Load the representations from the pickle file
        with open(self.config.represent_file, "rb") as f:
            representations = pickle.load(f)

        # Check if the representations are out-of-date
        if len(representations) > 0:
            if len(representations[0]) != len(df_cols):
                raise ValueError(
                    f"Seems existing {self.config.represent_file} is out-of-the-date."
                    "Please delete it and re-run."
                )
            pickled_images = [representation[0] for representation in representations]
        else:
            pickled_images = []

        # Get the list of images on storage
        storage_images =self.list_of_image_path(path=self.config.train_faces_dir)

        # Enforce data consistency amongst on disk images and pickle file
        must_save_pickle = False
        new_images = list(set(storage_images) - set(pickled_images)) # images added to storage
        old_images = list(set(pickled_images) - set(storage_images)) # images removed from storage

        if not self.config.silent and (len(new_images) > 0 or len(old_images) > 0):
            logger.info(f"Found {len(new_images)} new images and {len(old_images)} removed images")

        # remove old images first
        if len(old_images)>0:
            representations = [rep for rep in representations if rep[0] not in old_images]
            must_save_pickle = True

        # find representations for new images
        if len(new_images)>0:
            representations += self.find_bulk_embeddings(
                employees=new_images,
                model_name=self.config.model_name,
                target_size=target_size,
                detector_backend=self.config.face_detector_backend,
                enforce_detection=self.config.enforce_detection,
                align=self.config.align,
                normalization=self.config.normalization,
                silent=self.config.silent,
            ) # add new images
            must_save_pickle = True

        if must_save_pickle:
            with open(self.config.represent_file, "wb") as f:
                pickle.dump(representations, f)
            if not self.config.silent:
                logger.info(f"There are now {len(representations)} representations in representation.pkl")

        # Should we have no representations bailout
        if len(representations) == 0:
            if not self.config.silent:
                toc = time.time()
                logger.info(f"find function duration {toc - tic} seconds")
            return []
