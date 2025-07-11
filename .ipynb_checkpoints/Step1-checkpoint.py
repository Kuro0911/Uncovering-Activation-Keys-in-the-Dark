import torch
import math
import random
import os
import json
import numpy as np
from diffusers import DiffusionPipeline, AutoPipelineForText2Image, StableDiffusionXLPipeline, StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from datasets import load_dataset
import nltk
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import torch.nn.functional as F
from nltk.tag import pos_tag
from nltk.corpus import words, wordnet, brown, stopwords
from sklearn.decomposition import PCA
from config import DEVICE, POPULATION_SIZE, MAX_PROMPT_LENGTH, SIMILARITY_TOP_N, NUM_IMAGES_PER_EVAL, SEEDS, LORA_SCALE, MAX_GENERATIONS, NUM_INFERENCE_STEPS, IMAGE_MODEL

nltk.download('words')
nltk.download('wordnet')
nltk.download('brown')
nltk.download('stopwords')

class PromptEvolution:
    def __init__(self, base_pipe, lora_pipe, clip_model, clip_processor, LORA_PATH):
        self.base_pipe = base_pipe
        self.lora_pipe = lora_pipe
        # self.vocab = self.load_vocab_from_nltk()
        self.clip_model = clip_model
        self.clip_processor = clip_processor

        self.filtered_vocab = self.load_vocab_from_nltk()
        self.word_frequencies = self.get_word_frequencies()

        self.metadata_vocab = self.load_vocab_from_metadata(LORA_PATH)

        self.vocab_embeddings = self.precompute_embeddings()
        self.population = self.initialize_population()
        self.word_scores = defaultdict(float)
        self.prompt_cache = {}

        self.word_scores = defaultdict(float)
        self.prompt_history = defaultdict(float)
        self.all_time_best = {'score': -np.inf, 'prompt': ''}

        self.pca = PCA(n_components=32)
        self.vocab_embeddings_pca = self.pca.fit_transform(
            self.vocab_embeddings)

        self.cache_file = 'prompt_evolution_cache.json'
        self.clear_cache()

    def clear_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'w') as f:
                json.dump({"prompt_history": {}, "word_scores": {},
                          "all_time_best": {"score": -np.inf, "prompt": ""}}, f)
            print("Cache file cleared. Starting fresh.")

    def save_cache(self):
        data = {
            'prompt_history': dict(self.prompt_history),
            'word_scores': dict(self.word_scores),
            'all_time_best': self.all_time_best
        }
        with open(self.cache_file, 'w') as f:
            json.dump(data, f)

    def load_vocab_from_nltk(self):
        brown_words = brown.words()
        word_frequencies = Counter(word.lower() for word in brown_words)

        valid_words = set(words.words())
        pos_vocab = []
        for word in valid_words:
            synsets = wordnet.synsets(word)
            if synsets:
                pos = synsets[0].pos()
                if pos in ['n', 'a']:
                    if word_frequencies.get(word.lower(), 0) > 10:
                        pos_vocab.append(word)
        return pos_vocab

    def load_vocab_from_metadata(self, file_path: str) -> list:
        try:
            with open(file_path, 'rb') as f:
                header_length_bytes = f.read(8)
                header_length = int.from_bytes(
                    header_length_bytes, byteorder='little')
                header_json_bytes = f.read(header_length)

            header_data = json.loads(header_json_bytes.decode('utf-8'))
            tag_freq = header_data.get(
                "__metadata__", {}).get("ss_tag_frequency", {})
            data = json.loads(tag_freq)
            inner_dict = next(iter(data.values()))
            tag_list = [(phrase.strip(), freq)
                        for phrase, freq in inner_dict.items()]
            tag_list.sort(key=lambda x: x[1], reverse=True)

            filler_words = set(stopwords.words('english'))
            processed = defaultdict(float)
            for phrase, score in tag_list:
                words_in_phrase = phrase.split()
                filtered_words = [
                    w for w in words_in_phrase if w.lower() not in filler_words]
                if filtered_words:
                    per_word_score = score / len(filtered_words)
                    for w in filtered_words:
                        processed[w.lower()] += per_word_score

            adjusted = {}
            for word, score in processed.items():
                adjusted[word] = score / \
                    (self.word_frequencies.get(word, 0) + 1)

            processed_list = sorted(
                adjusted.items(), key=lambda x: x[1], reverse=True)

            print("Metadata vocabulary (word, score):", processed_list)
            return processed_list
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []

    def get_word_frequencies(self):
        brown_words = brown.words()
        return Counter(word.lower() for word in brown_words)

    def compute_embedding(self, token: str):
        try:
            token_id = self.base_pipe.tokenizer.convert_tokens_to_ids(token)
            inputs = torch.tensor(
                [token_id], dtype=torch.long).unsqueeze(1).to(DEVICE)
            with torch.no_grad():
                embedding = self.base_pipe.text_encoder(
                    inputs).last_hidden_state
            # Extract the CLS token representation and return as 1D array.
            return embedding[:, 0, :].cpu().numpy()[0]
        except Exception as e:
            print(f"Error computing embedding for token '{token}': {e}")
            return None

    def precompute_embeddings(self):
        token_ids = [self.base_pipe.tokenizer.convert_tokens_to_ids(
            t) for t in self.filtered_vocab]

        inputs = torch.tensor(token_ids, dtype=torch.long).unsqueeze(
            1).to(DEVICE)  # Ensure batch dimension

        with torch.no_grad():
            embeddings = self.base_pipe.text_encoder(
                inputs).last_hidden_state  # Use last_hidden_state
            # Extract CLS token representation
            return embeddings[:, 0, :].cpu().numpy()

    def initialize_population(self):
        return [self.random_combination() for _ in range(POPULATION_SIZE)]

    def random_combination(self):
        prompt_length = random.randint(1, MAX_PROMPT_LENGTH)
        sorted_vocab = sorted(
            self.filtered_vocab, key=lambda w: -self.word_frequencies.get(w.lower(), 0))

        if self.metadata_vocab:
            meta_words, meta_scores = zip(*self.metadata_vocab)
            meta_words = list(meta_words)
            meta_scores = list(meta_scores)
        else:
            meta_words, meta_scores = [], []

        tokens = []
        for _ in range(prompt_length):
            candidate = None
            available_meta = [w for w in meta_words if w not in tokens]
            available_weights = [
                meta_scores[meta_words.index(w)] for w in available_meta]
            if available_meta and random.random() < 0.4:
                tokens.append(random.choices(available_meta,
                              weights=available_weights, k=1)[0])
            else:
                tokens.append(random.choice(sorted_vocab))
        random.shuffle(tokens)
        return ",".join(tokens)

    def update_word_scores(self, prompt, score):
        tokens = [w.strip() for w in prompt.split(",") if w.strip()]
        prompt_length = len(tokens)
        if prompt_length == 0:
            return

        length_weight = 1 / math.sqrt(prompt_length)
        modulated_score = score * length_weight

        if modulated_score < 0:
            modulated_score *= 1.2

        for token in tokens:
            self.word_scores[token] += modulated_score

    def get_word_score(self, word):
        return self.word_scores.get(word, 0)

    def get_similar_words(self, target_word, top_n=50, exploration=0.3):
        try:
            if target_word in self.filtered_vocab:
                idx = self.filtered_vocab.index(target_word)
                target_embed = self.vocab_embeddings_pca[idx]
            else:
                # Compute embedding for the missing word.
                computed_embed = self.compute_embedding(target_word)
                if computed_embed is None:
                    # Fallback: return a random sample from filtered_vocab
                    return random.sample(self.filtered_vocab, top_n)
                # Transform the computed embedding via PCA
                target_embed = self.pca.transform(
                    computed_embed.reshape(1, -1))[0]

            similarities = cosine_similarity(
                [target_embed], self.vocab_embeddings_pca)[0]
            target_score = self.get_word_score(target_word)
            sorted_indices = np.argsort(-similarities)
            all_candidates = [self.filtered_vocab[i] for i in sorted_indices]

            if target_score >= 0:
                split_idx = int(len(all_candidates) * 0.9)
                similar = all_candidates[:split_idx]
                dissimilar = all_candidates[split_idx:]
                selected = similar[:int(top_n*0.9)] + \
                    dissimilar[-int(top_n*0.1):]
            else:
                split_idx = int(len(all_candidates) * 0.1)
                similar = all_candidates[:split_idx]
                dissimilar = all_candidates[split_idx:]
                selected = dissimilar[:int(top_n*0.9)] + \
                    similar[-int(top_n*0.1):]

            if random.random() < exploration:
                random.shuffle(selected)

            return selected[:top_n]

        except Exception as e:
            print(f"Error in get_similar_words for token '{target_word}': {e}")
            return random.sample(self.filtered_vocab, top_n)

    def mutate(self, prompt, fitness_score):
        tokens = [w.strip() for w in prompt.split(",") if w.strip()]
        operation_weights = {
            'add': 0.5 + (0.3 * (1 - fitness_score)),
            'remove': 0.3 * fitness_score,
            'replace': 0.4 + (0.3 * (1 - fitness_score))
        }

        total = sum(operation_weights.values())
        operation_probs = {k: v/total for k, v in operation_weights.items()}

        for _ in range(1 + int(2 * (1 - fitness_score))):
            operation = random.choices(
                list(operation_probs.keys()),
                weights=list(operation_probs.values())
            )[0]

            if operation == 'add' and len(tokens) < MAX_PROMPT_LENGTH:
                candidates = self.get_addition_candidates(
                    tokens, fitness_score)
                new_tok = self.select_word(candidates, fitness_score)
                if new_tok not in tokens:
                    tokens.append(new_tok)

            elif operation == 'remove' and tokens:
                self.remove_weak_words(tokens, fitness_score)

            elif operation == 'replace' and tokens:
                self.replace_weak_words(tokens, fitness_score)

        return ",".join(list(dict.fromkeys(tokens))[:MAX_PROMPT_LENGTH])

    def get_addition_candidates(self, existing_tokens, fitness_score):
        if existing_tokens and random.random() < 0.7:
            base_word = max(existing_tokens, key=self.get_word_score)
            candidates = self.get_similar_words(
                base_word,
                top_n=SIMILARITY_TOP_N,
                exploration=max(0.2, 0.6 - fitness_score)
            )
        else:
            candidates = sorted(self.filtered_vocab,
                                key=lambda w: -self.get_word_score(w))[:200]
        return candidates

    def select_word(self, candidates, fitness_score):
        if not candidates:
            return random.choice(self.filtered_vocab)

        scores = np.array([self.get_word_score(w) for w in candidates])
        scores = np.exp(scores - np.max(scores))
        scores /= scores.sum()

        if random.random() < 0.8 * fitness_score:
            return np.random.choice(candidates, p=scores)
        return random.choice(candidates)

    def remove_weak_words(self, tokens, fitness_score):
        if len(tokens) > 1:
            scores = [self.get_word_score(t) for t in tokens]
            threshold = np.percentile(
                scores, 30 if fitness_score > 0.5 else 50)
            weak_indices = [i for i, s in enumerate(scores) if s < threshold]
            if weak_indices:
                tokens.pop(random.choice(weak_indices))

    def replace_weak_words(self, tokens, fitness_score):
        scores = [self.get_word_score(t) for t in tokens]
        replace_idx = np.argmin(scores)
        old_word = tokens[replace_idx]

        candidates = self.get_similar_words(
            old_word,
            top_n=SIMILARITY_TOP_N,
            exploration=0.8 if fitness_score < 0.4 else 0.3
        )
        new_word = self.select_word(candidates, fitness_score)
        tokens[replace_idx] = new_word

    def crossover(self, parent1, parent2):
        tokens1 = parent1.split(",")
        tokens2 = parent2.split(",")
        combined = []
        for t1 in tokens1:
            if t1 in tokens2:
                combined.append(t1)
            else:
                similar = self.get_similar_words(t1, top_n=20)
                matches = [t for t in similar if t in tokens2]
                if matches:
                    combined.append(random.choice(matches))
        remaining = list(set(tokens1 + tokens2) - set(combined))
        random.shuffle(remaining)
        combined += remaining[:MAX_PROMPT_LENGTH - len(combined)]
        return ",".join(combined)
    
    def show_images(self, prompt_str, score, base_images, lora_images, generation, output_dir: str = "results"):
        gen_folder = os.path.join(output_dir, f"generation_{generation}")
        os.makedirs(gen_folder, exist_ok=True)

        fig, axes = plt.subplots(2, NUM_IMAGES_PER_EVAL, figsize=(15, 6))
        fig.suptitle(f"Gen {generation} | Prompt: {prompt_str} | Score: {score:.2f}", fontsize=12)
    
        for i in range(NUM_IMAGES_PER_EVAL):
            axes[0, i].imshow(base_images[i])
            axes[0, i].axis("off")
            axes[1, i].imshow(lora_images[i])
            axes[1, i].axis("off")

        slug = re.sub(r'[^0-9A-Za-z]+', '_', prompt_str)   
        slug = re.sub(r'_+', '_', slug)                  
        slug = slug.strip('_').lower()                    
        if len(slug) > 80:
            slug = slug[:80].rstrip('_')
            
        filename = f"{slug}.png"
        save_path = os.path.join(gen_folder, filename)
    
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    def evaluate_prompt(self, prompt, generation):
        print("-"*40 + f" Evaluating Prompt: {prompt} " + "-"*40)
        cached_score = self.prompt_history.get(prompt, None)
        if cached_score is not None:
            print(f"Found in cache: {prompt} score: {cached_score}")
            return cached_score

        try:
            base_images, lora_images = [], []
            tokens = [w.strip() for w in prompt.split(",") if w.strip()]
            prompt_str = ",".join(tokens)
            for seed in SEEDS:
                generator = torch.Generator(DEVICE).manual_seed(seed)
                
                if IMAGE_MODEL == "stabilityai/stable-diffusion-xl-base-1.0":
                    base_images.append(self.base_pipe(prompt_str, num_inference_steps=num_inference_steps, generator=generator).images[0])
                    lora_images.append(self.lora_pipe(prompt_str, num_inference_steps=num_inference_steps, generator=generator, cross_attention_kwargs={"scale": LORA_SCALE}).images[0])
                elif IMAGE_MODEL == "sd-legacy/stable-diffusion-v1-5":
                    (prompt_embeds, negative_prompt_embeds) = self.base_pipe.encode_prompt(
                        prompt_str, DEVICE, num_images_per_prompt=1, do_classifier_free_guidance=True)
                    base_images.append(self.base_pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                                       num_inference_steps=NUM_INFERENCE_STEPS, generator=generator).images[0])
                    lora_images.append(self.lora_pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                                       num_inference_steps=NUM_INFERENCE_STEPS, generator=generator, cross_attention_kwargs={"scale": LORA_SCALE}).images[0])
            
            intra_model_consistency_1, base_features = self.calculate_similarity(base_images)
            intra_model_consistency_2, lora_features = self.calculate_similarity(lora_images)

            sim_matrix = base_features @ lora_features.T
            inter_model_similarity = sim_matrix.mean().item()

            base_spread = self.calculate_spread(base_images)
            lora_spread = self.calculate_spread(lora_images)

            print(f"intra_model_consistency_1: {intra_model_consistency_1}")
            print(f"intra_model_consistency_2: {intra_model_consistency_2}")
            print(f"inter_model_similarity: {inter_model_similarity}")
            print(f"base_spread: {base_spread}")
            print(f"lora_spread: {lora_spread}")

            alpha = 1.5
            beta = 1
            gamma = 1.3

            score = -alpha * lora_spread + beta * \
                (1 - inter_model_similarity) + gamma * base_spread
            score = np.tanh(score)

            # score = -lora_spread + (1 - inter_model_similarity + base_spread)

            for token in tokens:
                self.word_scores[token] += score
                
            self.show_images(prompt_str, score, base_images, lora_images, generation)

            if score > self.all_time_best['score']:
                self.all_time_best = {'score': score, 'prompt': prompt}

            self.update_word_scores(prompt, score)

            self.prompt_history[prompt] = max(
                score, self.prompt_history.get(prompt, -np.inf))

            return score
        except Exception as e:
            print(e)
            return float("-inf")

    def calculate_similarity(self, images):
        inputs = self.clip_processor(
            images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        features = F.normalize(features, dim=-1)
        sim_matrix = features @ features.T
        return sim_matrix.mean().item(), features

    def calculate_spread(self, images):
        inputs = self.clip_processor(
            images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        features = F.normalize(features, dim=-1)
        pairwise_dists = pdist(features.cpu().numpy(), metric='cosine')
        return np.mean(pairwise_dists)

    def run_evolution(self):
        best_score = float("-inf")
        best_prompt = ""
        diversity_history = []
        crossover_prob = 0.05  # Small probability of crossover

        try:
            for generation in range(MAX_GENERATIONS):
                print( "#"*40 + f"GENERATION : {generation + 1} / {MAX_GENERATIONS} " + "#"*40)
                print("current pool: ")
                print(self.population)

                scores = [self.evaluate_prompt(p, generation) for p in self.population]

                unique_prompts = len(set(self.population))
                diversity_history.append(unique_prompts)
                print(f"Diversity: {unique_prompts}/{len(self.population)}")

                current_best_idx = np.nanargmax(scores)
                current_best_score = scores[current_best_idx]
                if current_best_score > best_score:
                    best_score = current_best_score
                    best_prompt = self.population[current_best_idx]

                sorted_pop = [p for _, p in sorted(
                    zip(scores, self.population), key=lambda x: x[0], reverse=True)]
                elite_size = int(POPULATION_SIZE * 0.2)
                elite = sorted_pop[:elite_size]

                new_pop = elite.copy()

                new_pop += [self.random_combination()
                            for _ in range(int(POPULATION_SIZE * 0.1))]

                while len(new_pop) < POPULATION_SIZE:
                    if random.random() < crossover_prob:
                        parent1 = random.choice(elite)
                        parent2 = random.choice(self.population)
                        child = self.crossover(parent1, parent2)
                    else:
                        if random.random() < 0.3:
                            parent = random.choice(elite)
                        else:
                            parent = random.choice(self.population)
                        parent_score = scores[self.population.index(parent)]
                        child = self.mutate(parent, parent_score)
                    new_pop.append(child)

                if len(set(new_pop)) < POPULATION_SIZE * 0.5:
                    print("Applying diversity boost")
                    new_pop = list(set(new_pop))
                    new_pop += [self.random_combination()
                                for _ in range(POPULATION_SIZE - len(new_pop))]

                self.population = new_pop[:POPULATION_SIZE]

                if generation % 5 == 0:
                    print("Resetting word scores")
                    self.word_scores = defaultdict(float)
                    if generation % 5 == 0:
                        self.save_cache()

            return self.all_time_best['prompt'], self.all_time_best['score']
        finally:
            self.save_cache()
