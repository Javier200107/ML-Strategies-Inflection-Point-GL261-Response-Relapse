import numpy as np
import SimpleITK as sitk
import radiomics

class FeatureExtractorRadiomics:
    def __init__(self, image=None, mask=None):
        self.image = None
        self.mask = None
        if image is not None and mask is not None:
            self.set_image_and_mask(image, mask)

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

    # Function to apply image filters and extract features
    def extract_filtered_features(self, filter_type='wavelet'):
        filters = {
            'wavelet': radiomics.imageoperations.getWaveletImage,
            'logarithm': radiomics.imageoperations.getLogarithmImage,
            'square': radiomics.imageoperations.getSquareImage,
            'squareroot': radiomics.imageoperations.getSquareRootImage,
            'exponential': radiomics.imageoperations.getExponentialImage,
            'gradient': radiomics.imageoperations.getGradientImage,
            'lbp2D': radiomics.imageoperations.getLBP2DImage,
        }

        if filter_type in filters:
            filtered_images = filters[filter_type](self.image, self.mask)
            feature_dict = {}
            # Iterate over the generator returned by the filter function
            for filtered_image, filtered_image_Name, kwargs in filtered_images:
                # Temporarily update self.image to the filtered image
                original_image = self.image
                self.image = filtered_image

                # Extract features from the filtered image (you can call multiple feature extractors here)
                features = self.extract_first_order_features()
                feature_dict.update({f"{filter_type}_{k}": v for k, v in features.items()})
                
                # Optionally extract other feature classes, like GLCM, GLRLM, etc.
                features = self.extract_glcm_features()
                feature_dict.update({f"{filter_type}_{k}": v for k, v in features.items()})

                features = self.extract_glrlm_features()
                feature_dict.update({f"{filter_type}_{k}": v for k, v in features.items()})

                features = self.extract_glszm_features()
                feature_dict.update({f"{filter_type}_{k}": v for k, v in features.items()})

                features = self.extract_ngtdm_features()
                feature_dict.update({f"{filter_type}_{k}": v for k, v in features.items()})

                features = self.extract_gldm_features()
                feature_dict.update({f"{filter_type}_{k}": v for k, v in features.items()})

                # Reset self.image to original after extraction
                self.image = original_image

            return feature_dict
        else:
            raise ValueError(f"Filter type '{filter_type}' is not supported.")

    # Function to extract all radiomic features (without 3D features)
    def extract_all_radiomic_features(self):
        # Extract and combine features from the original image
        feature_dicts = []
        feature_dicts.append(self.extract_first_order_features())
        feature_dicts.append(self.extract_shape_2d_features())
        feature_dicts.append(self.extract_glcm_features())
        feature_dicts.append(self.extract_glrlm_features())
        feature_dicts.append(self.extract_glszm_features())
        feature_dicts.append(self.extract_ngtdm_features())
        feature_dicts.append(self.extract_gldm_features())

        # Apply filters and extract features from filtered images
        filters = ['wavelet', 'logarithm', 'square', 'squareroot', 'exponential', 'gradient', 'lbp2D']
        for filter_type in filters:
            filtered_features = self.extract_filtered_features(filter_type)
            feature_dicts.append(filtered_features)

        # Combine all feature dictionaries into a single one
        combined_features = {}
        for feature_dict in feature_dicts:
            combined_features.update(feature_dict)

        return combined_features
