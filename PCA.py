from sklearn.decomposition import PCA
from get_data_and_define_functions import x_train_simple, x_test_simple, test_samples, encoding_dim
from get_data_and_define_functions import plot_examples, mean_sq_er

# applying PCA to reduce dimensions of features to 3
pca = PCA(encoding_dim).fit(x_train_simple)

print("Principal axes in feature space: ", pca.components_, "\n")
print("The amount of variance explained by each of the selected components.: ", pca.explained_variance_)

# applying PCA transformation to test set
z = pca.transform(x_test_simple)

# inverse trasnformation application, maybe equivalent to decoding operation on AE
final = pca.inverse_transform(z)

print("MSE over the validation set...", mean_sq_er(x_test_simple, final, test_samples))

plot_examples(x_test_simple, final, 3)
