def initiate_data_transformation(self, train_path, test_path):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logging.info(f"Train data shape: {train_df.shape}")
        logging.info(f"Test data shape: {test_df.shape}")

        if train_df.shape[0] < 2:
            raise CustomException("Train data is insufficient", sys)

        target_column_name = 'Machine failure'
        if target_column_name not in train_df.columns:
            raise CustomException(f"Target column '{target_column_name}' not found in train data", sys)
        if target_column_name not in test_df.columns:
            raise CustomException(f"Target column '{target_column_name}' not found in test data", sys)

        # Separate input features and target for train data
        input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
        target_feature_train_df = train_df[target_column_name]

        # Separate input features and target for test data
        input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
        target_feature_test_df = test_df[target_column_name]

        # Automatically infer numerical and categorical columns from train features
        numerical_cols = input_feature_train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = input_feature_train_df.select_dtypes(include=['object', 'category']).columns.tolist()

        preprocessor = self.get_data_transformer_object(numerical_cols, categorical_cols)

        # Check for missing columns in test features
        missing_cols = [col for col in numerical_cols + categorical_cols if col not in input_feature_test_df.columns]
        if missing_cols:
            raise CustomException(f"Test data missing columns: {missing_cols}", sys)

        # Transform features
        input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessor.transform(input_feature_test_df)

        # Combine features and target to create train and test arrays
        train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]  # <-- labels included here

        # Save the preprocessor object for future use
        save_object(
            file_path=self.data_transformation_config.preprocessed_object_file_path,
            obj=preprocessor
        )

        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessed_object_file_path
        )

    except Exception as e:
        logging.info("Error in initiate_data_transformation")
        raise CustomException(e, sys)
