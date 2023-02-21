import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class ProductReviewDataset(Dataset):
    """MovieLens PyTorch Dataset for Training

    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds

    """

    def __init__(self,
                 data_path: str = '/home/amirhossein/PycharmProjects/Recommender/dataset/amazon_reviews_us_Electronics_v1_00.xlsx',
                 phase: str = 'train'):
        df = pd.read_excel(data_path, nrows=999999)
        df = df[['customer_id', 'product_id', 'star_rating']]
        # Filter long tail
        customers = df['customer_id'].value_counts()
        products = df['product_id'].value_counts()

        customers = customers[customers >= 5]
        products = products[products >= 6]

        reduced_df = df.merge(pd.DataFrame({'customer_id': customers.index})).merge(
            pd.DataFrame({'product_id': products.index}))

        customers = reduced_df['customer_id'].value_counts()
        products = reduced_df['product_id'].value_counts()

        self.customer_index = pd.DataFrame({'customer_id': customers.index, 'user': np.arange(customers.shape[0])})
        self.product_index = pd.DataFrame({'product_id': products.index, 'item': np.arange(products.shape[0])})
        reduced_df = reduced_df.merge(self.customer_index).merge(self.product_index)
        if phase == 'train':
            self.data_df, _ = train_test_split(reduced_df, test_size=0.2, random_state=0)
        else:
            _, self.data_df = train_test_split(reduced_df, test_size=0.2, random_state=0)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        user = self.data_df['user'].values[idx]
        item = self.data_df['item'].values[idx]
        star_rating = self.data_df['star_rating'].values[idx]
        return {'user': user, 'item': item, 'star_rating': star_rating}


if __name__ == '__main__':
    data = ProductReviewDataset()

    for item in data:
        print(item)
