from collections import Counter, defaultdict
import csv
import random
from typing import List, Tuple, Dict, NamedTuple

import tqdm

from dsfs.nlp import cosine_similarity
from dsfs.deep_learning import random_tensor
from dsfs.linalg.vector import dot


def popular_interests(user_interests):
    return Counter(interest for user_interest in user_interests for interest in user_interest)


def most_popular_new_interests(
    user_interests: List[str], popular_interests, max_results: int = 5
) -> List[Tuple[str, int]]:
    suggestions = [
        (interest, frequency)
        for interest, frequency in popular_interests.most_common()
        if interest not in user_interests
    ]
    return suggestions[:max_results]


def get_unique_interests(users_interests):
    return sorted({interest for user_interests in users_interests for interest in user_interests})


def make_user_interest_vector(user_interests: List[str], unique_interests) -> List[int]:
    return [1 if interest in user_interests else 0 for interest in unique_interests]


def get_user_similarities(user_interest_vectors):
    return [
        [
            cosine_similarity(interest_vector_i, interest_vector_j)
            for interest_vector_j in user_interest_vectors
        ]
        for interest_vector_i in user_interest_vectors
    ]


def most_similar_users_to(user_id: int, user_similarities) -> List[Tuple[int, float]]:
    pairs = [
        (other_user_id, similarity)
        for other_user_id, similarity in enumerate(user_similarities[user_id])
        if user_id != other_user_id and similarity > 0
    ]

    return sorted(pairs, key=lambda pair: pair[-1], reverse=True)


def user_based_suggestions(
    user_id: int, user_similarities, users_interests, include_current_interests: bool = False
):
    suggestions: Dict[str, float] = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_id, user_similarities):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity

    suggestions = sorted(suggestions.items(), key=lambda pair: pair[-1], reverse=True)

    if include_current_interests:
        return suggestions
    else:
        return [
            (suggestion, weight)
            for suggestion, weight in suggestions
            if suggestion not in users_interests[user_id]
        ]


def transpose(unique_interests, user_interest_vectors):
    return [
        [user_interest_vector[j] for user_interest_vector in user_interest_vectors]
        for j, _ in enumerate(unique_interests)
    ]


def get_interests_similarities(interest_user_matrix):
    return [
        [cosine_similarity(user_vector_i, user_vector_j) for user_vector_j in interest_user_matrix]
        for user_vector_i in interest_user_matrix
    ]


def most_similar_interests_to(interest_id: int, interest_similarities, unique_interests):
    similarities = interest_similarities[interest_id]
    pairs = [
        (unique_interests[other_interest_id], similarity)
        for other_interest_id, similarity in enumerate(similarities)
        if interest_id != other_interest_id and similarity > 0
    ]
    return sorted(pairs, key=lambda pair: pair[-1], reverse=True)


def item_based_suggestions(user_id: int, user_interest_vectors, interest_similarities, unique_interests, user_interests, include_current_interests: bool = False):
    suggestions = defaultdict(float)
    user_interest_vector = user_interest_vectors[user_id]
    for interest_id, is_interested in enumerate(user_interest_vector):
        if is_interested == 1:
            similar_interests = most_similar_interests_to(interest_id, interest_similarities, unique_interests)
            for interest, similarity in similar_interests:
                suggestions[interest] += similarity

    suggestions = sorted(suggestions.items(), key=lambda pair: pair[-1], reverse=True)

    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight) 
                for suggestion, weight in suggestions
                if suggestion not in user_interests[user_id]
        ]


MOVIES = "u.item"
RATINGS = "u.data"


class Rating(NamedTuple):
    user_id: str
    movie_id: str
    rating: float


def get_movie_data():
    with open(MOVIES, encoding="iso-8859-1") as f:
        reader = csv.reader(f, delimiter="|")
        return {movie_id: title for movie_id, title, *_ in reader}


def get_ratings_data():
    with open(RATINGS, encoding="iso-8859-1") as f:
        reader = csv.reader(f, delimiter="\t")
        return [Rating(user_id, movie_id, float(rating))
                for user_id, movie_id, rating, _ in reader]


def split_data(ratings):
    random.shuffle(ratings)

    split1 = int(len(ratings) * 0.7)
    split2 = int(len(ratings) * 0.85)

    return ratings[:split1], ratings[split1:split2], ratings[split2:]


def baseline_error(train, test):
    avg_rating = sum(rating.rating for rating in train) / len(train)
    return sum((rating.rating - avg_rating) ** 2 for rating in test) / len(test)


EMBEDDING_DIM = 2


def create_embeddings(ratings):
    user_ids = {rating.user_id for rating in ratings}
    movie_ids = {rating.movie_id for rating in ratings}

    user_vectors = {user_id: random_tensor(EMBEDDING_DIM) for user_id in user_ids}
    movie_vectors = {movie_id: random_tensor(EMBEDDING_DIM) for movie_id in movie_ids}

    return user_vectors, movie_vectors


def loop(dataset: List[Rating], movie_vectors, user_vectors, learning_rate: float = None) -> None:
    with tqdm.tqdm(dataset) as t:
        loss = 0.0
        for i, rating in enumerate(t):
            movie_vector = movie_vectors[rating.movie_id]
            user_vector = user_vectors[rating.user_id]
            predicted = dot(user_vector, movie_vector)
            error = predicted - rating.rating
            loss += error ** 2

            if learning_rate is not None:
                user_gradient = [error * m_j for m_j in movie_vector]
                movie_gradient = [error * u_j for u_j in user_vector]

                for j in range(EMBEDDING_DIM):
                    user_vector[j] -= learning_rate * user_gradient[j]
                    movie_vector[j] -= learning_rate * movie_gradient[j]
            t.set_description(f"avg loss: {loss / (i + 1)}")


def train(ratings):
    user_vectors, movie_vectors = create_embeddings(ratings)

    train, validation, test = split_data(ratings)

    learning_rate = 0.05
    for epoch in range(20):
        learning_rate *= 0.9
        print(epoch, learning_rate)
        loop(train, movie_vectors, user_vectors, learning_rate=learning_rate)
        loop(validation, movie_vectors, user_vectors)
    loop(test, movie_vectors, user_vectors)
