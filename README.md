# Matrix Factorization

Matrix factorization is a simple embedding model. Given the feedback matrix $A \in R^{m \times n}$, where m is the number of users (or queries) and n is the number of items, the model learns :
- A user embedding matrix $U \in R^{m \times d}$, where row i is the embedding for user i.
- An item embedding matrix $V \in R^{n \times d}$, where row j is the embedding for item j.
 
![image](https://github.com/user-attachments/assets/ba604211-04ee-46aa-b7c1-f0b386e1921d)

The embeddings are learned such that the product $UV^T$ is a good approximation of feedback matrix A. Observe that the (i,j)  entry of $U.V^T$  is simply the dot product $U_i.V^T_j$ of the embeddings of user i and 
item j, which you want to be close to $A_{i,j}$ .  The Feature learned in this method is called **Latent Feature**.

 **Note:** Matrix factorization typically gives a more compact representation than learning the full matrix.
 The full matrix has  $O(mn)$  entries, while the embedding matrices $U,V$ have $O((m+n)d$ entries,
 where the embedding dimension d is typically much smaller than m and n  . As a result, matrix
 factorization finds latent structure in the data, assuming that observations lie close to a low-dimensional
 subspace. In the preceding example, the values of n, m, and d are so low that the advantage is negligible.
 In real-world recommendation systems, however, matrix factorization can be signi cantly more compact
 than learning the full matrix.

 ### Choosing the Objective Function
 One intuitive objective function is the squared distance. To do this, minimize the sum of
 squared errors over all pairs of observed entries:

 $$
 \underset{U \in R^{m \times d} , V \in R^{n \times d}}{min} \sum_{(i,j) \in Obs} (A_{i,j} - U_i.V^T_j)^2
 $$

In this objective function, you only sum over observed pairs (i, j), that is, over non-zero
 values in the feedback matrix. However, only summing over values of one is not a good idea
 —a matrix of all ones will have a minimal loss and produce a model that can't make
 effective recommendations and that generalizes poorly.

 ![image](https://github.com/user-attachments/assets/fcfd1381-7655-4af8-86f3-d76ae481f169)

Perhaps you could treat the unobserved values as zero, and sum over all entries in the matrix. This corresponds to minimizing the squared 
Frobenius distance between  A and its approximation $UV^T$ :

$$
\underset{U \in R^{m \times d} , V \in R^{n \times d}}{min} ||A-UV^T||^2_F
$$

You can solve this quadratic problem through **Singular Value Decomposition (SVD) of the
 matrix**. However, SVD is not a great solution either, because in real applications, the matrix A
 may be very sparse. For example, think of all the videos on YouTube compared to all the
 videos a particular user has viewed. The solution $UV^T$ (which corresponds to the model's
 approximation of the input matrix) will likely be close to zero, leading to poor generalization
 performance.

 In contrast, Weighted Matrix Factorization decomposes the objective into the following two
 sums:
 A sum over observed entries.
 A sum over unobserved entries (treated as zeroes)

 $$
\underset{U \in R^{m \times d} , V \in R^{n \times d}}{min} \sum_{(i,j) \in obs} (A_{i,j} - U_i.V^T_j)^2 + w_0 \sum_{(i,j) \notin obs} (U.V^T)^2
$$

Here, $w_0$ is a hyperparameter that weights the two terms so that the objective is not
 dominated by one or the other. Tuning this hyperparameter is very important.

 #### In Summery : 
 In recommendation system , we use users past bahaviour and recommend the movies of their intrests. The story begins with **Content Based Filtering** where we recommend movies based on geners like  Comedy , Action, 
 Drama , Horror, Name of Director , Actors,and so many things   ....etc but peoples mood changes over time and in this method we need more and more feature to recommend movies , this method won't work effectively.
 Now we have another method called **Collaborative Filtering** where we take user's preferance data for example ratings and geners like this
 
 ![image](https://github.com/user-attachments/assets/2cc63e1f-88e6-4474-84a1-d0e4b2a67e58)

 Netflix started it did just this it would ask new users to fill out a laundry list of questions about their preferences before presenting them with suggested movies and this leads to the problem of having to 
 collect all of this preference data on users not only is it a burden for users it's also prone to failure as we aren't always great at describing our own preferences sometimes we simply can't explain why we 
 like things.

 ##### Now we generate the feature from user ratings data 
 Now we  started  throw away the idea of dreaming up features used to connect peoples and movies instead we flip things around and use the user preference data we do have to generate the features. for example we might 
 have this incomplete set of preference data and we will instead learn or discover the relevant features based on patterns in this data and this is done by simply reversing the problem. We first perform an 
 approximate factorization into two matrices and we can do this using a machine learning approach. The job of the machine learning algorithm is to guess values for those matrices which will match the existing 
 data and the preference matrix as closely as possible .The simplest approach is to simply guess numbers over and over until you arrive at a set of numbers which predict the data with the lowest air overall once 
 this estimation is finished we can multiply the matrices as before to fill in all of the missing values.  It's important to note that we won't know exactly what to label these discovered features so we call them **latent
 features** because they arise out of the underlying patterns and the data you can think of them as an average or weighted sum of the patterns in the data they are not based on a human defined feature such as comedy and that's 
 the key insight behind this method with content filtering , the features come from the human mind whereas with collaborative filtering the, features are extracted directly from the patterns in the data and this will predict the 
 data in the same way.

 ![image](https://github.com/user-attachments/assets/c34ea57e-4f6d-4781-a36b-acbf9fc7c112)

 #### Acknowledgement : 
 - Google's Advance Machine Learning 


 
