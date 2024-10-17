import numpy as np
import SimpleITK as sitk
import radiomics

class FeatureExtractorRadiomics:
    def __init__(self, image=None, mask=None):
        self.image = None
        self.mask = None
        if image is not None and mask is not None:
            self.set_image_and_mask(image, mask)
        #self.features = None

    # Setter method to update the image and mask
    def set_image_and_mask(self, image, mask):
        self.image = sitk.GetImageFromArray(image)
        self.mask = sitk.GetImageFromArray(mask)

    # Function to extract first-order features
    def extract_first_order_features(self):
        extractor = radiomics.firstorder.RadiomicsFirstOrder(self.image, self.mask)
        return extractor.execute()

    # Function to extract 2D shape-based features
    def extract_shape_2d_features(self):
        image_2d = sitk.GetImageFromArray(np.squeeze(sitk.GetArrayFromImage(self.image)))
        mask_2d = sitk.GetImageFromArray(np.squeeze(sitk.GetArrayFromImage(self.mask)))

        extractor = radiomics.shape2D.RadiomicsShape2D(image_2d, mask_2d, force2D=True)
        return extractor.execute()

    # Functions to extract other 2D radiomic features
    def extract_glcm_features(self):
        extractor = radiomics.glcm.RadiomicsGLCM(self.image, self.mask)
        return extractor.execute()

    def extract_glrlm_features(self):
        extractor = radiomics.glrlm.RadiomicsGLRLM(self.image, self.mask)
        return extractor.execute()

    def extract_glszm_features(self):
        extractor = radiomics.glszm.RadiomicsGLSZM(self.image, self.mask)
        return extractor.execute()

    def extract_ngtdm_features(self):
        extractor = radiomics.ngtdm.RadiomicsNGTDM(self.image, self.mask)
        return extractor.execute()

    def extract_gldm_features(self):
        extractor = radiomics.gldm.RadiomicsGLDM(self.image, self.mask)
        return extractor.execute()

    # Function to extract all radiomic features (without 3D features)
    def extract_all_radiomic_features(self):

        # Extract and combine features from various categories
        feature_dicts = []
        feature_dicts.append(self.extract_first_order_features())
        feature_dicts.append(self.extract_shape_2d_features())
        feature_dicts.append(self.extract_glcm_features())
        feature_dicts.append(self.extract_glrlm_features())
        feature_dicts.append(self.extract_glszm_features())
        feature_dicts.append(self.extract_ngtdm_features())
        feature_dicts.append(self.extract_gldm_features())
        
        # Combine all feature dictionaries into a single one
        combined_features = {}
        for feature_dict in feature_dicts:
            combined_features.update(feature_dict)
        #self.features = combined_features
        return combined_features

    