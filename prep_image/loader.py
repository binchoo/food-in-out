import random
from tensorflow.keras.preprocessing import image_dataset_from_directory

def image_loader(directory, validation_split, 
                 image_size, batch_size, subset="all"):
    
    seed = random.randint(1, 150)
    
    if subset in ["train", "all"]: #prepare train_ds
        train_ds = image_dataset_from_directory(
            directory,
            labels='inferred',
            label_mode='categorical',
            validation_split=validation_split,
            subset="training",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size)
        
        if subset == "train": #return of "train" subset
            return train_ds
        
    if subset in ["valid", "all"]: #prepare valid_ds
        valid_ds = image_dataset_from_directory(
            directory,
            labels='inferred',
            label_mode='categorical',
            validation_split=validation_split,
            subset="validation",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size)
        
        if subset == "valid": #return of "valid" subset
            return valid_ds 
    
    else:
        raise Exception('Undefined subset:', subset)
                        
    return train_ds, valid_ds #return of "all" subset