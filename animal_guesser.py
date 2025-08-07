import json
import math
import os
from typing import Dict, List, Optional, Tuple
import pickle
from datetime import datetime

class AnimalGuesserML:
    def __init__(self, data_file='animal_data.json', model_file='ml_model.pkl'):
        self.data_file = data_file
        self.model_file = model_file
        self.animals = self.load_animals()
        self.questions = self.load_questions()
        self.game_history = self.load_game_history()
        self.stats = self.load_stats()
        
    def load_animals(self) -> List[Dict]:
        """Load animal database with features"""
        default_animals = [
            {"name": "Dog", "features": {"mammal": 1, "domestic": 1, "four_legs": 1, "barks": 1, "carnivore": 0.5}},
            {"name": "Cat", "features": {"mammal": 1, "domestic": 1, "four_legs": 1, "purrs": 1, "carnivore": 1}},
            {"name": "Elephant", "features": {"mammal": 1, "large": 1, "four_legs": 1, "trunk": 1, "herbivore": 1}},
            {"name": "Lion", "features": {"mammal": 1, "carnivore": 1, "four_legs": 1, "wild": 1, "roars": 1}},
            {"name": "Fish", "features": {"aquatic": 1, "fins": 1, "gills": 1, "scales": 1, "cold_blooded": 1}},
            {"name": "Bird", "features": {"flies": 1, "feathers": 1, "beak": 1, "lays_eggs": 1, "warm_blooded": 1}},
            {"name": "Snake", "features": {"reptile": 1, "no_legs": 1, "cold_blooded": 1, "carnivore": 1, "long": 1}},
            {"name": "Rabbit", "features": {"mammal": 1, "herbivore": 1, "four_legs": 1, "hops": 1, "long_ears": 1}},
            {"name": "Bear", "features": {"mammal": 1, "large": 1, "four_legs": 1, "omnivore": 1, "wild": 1}},
            {"name": "Whale", "features": {"mammal": 1, "aquatic": 1, "large": 1, "intelligent": 1, "warm_blooded": 1}}
        ]
        
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f).get('animals', default_animals)
        return default_animals
    
    def load_questions(self) -> List[Dict]:
        """Load question database with features and weights"""
        default_questions = [
            {"text": "Is it a mammal?", "feature": "mammal", "weight": 0.8},
            {"text": "Does it live on land?", "feature": "terrestrial", "weight": 0.7},
            {"text": "Is it larger than a house cat?", "feature": "large", "weight": 0.6},
            {"text": "Is it a carnivore (meat-eater)?", "feature": "carnivore", "weight": 0.7},
            {"text": "Does it have four legs?", "feature": "four_legs", "weight": 0.8},
            {"text": "Is it a domestic animal?", "feature": "domestic", "weight": 0.6},
            {"text": "Can it fly?", "feature": "flies", "weight": 0.9},
            {"text": "Does it live in water?", "feature": "aquatic", "weight": 0.8},
            {"text": "Is it a predator?", "feature": "predator", "weight": 0.6},
            {"text": "Does it have fur?", "feature": "fur", "weight": 0.7}
        ]
        
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f).get('questions', default_questions)
        return default_questions
    
    def load_game_history(self) -> List[Dict]:
        """Load previous game sessions for learning"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f).get('game_history', [])
        return []
    
    def load_stats(self) -> Dict:
        """Load game statistics"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f).get('stats', {"played": 0, "correct": 0})
        return {"played": 0, "correct": 0}
    
    def save_data(self):
        """Persist all data to JSON file"""
        data = {
            'animals': self.animals,
            'questions': self.questions,
            'game_history': self.game_history,
            'stats': self.stats
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def calculate_entropy(self, candidates: List[Dict], feature: str) -> float:
        """Calculate information entropy for feature selection"""
        if len(candidates) <= 1:
            return 0
        
        yes_count = sum(1 for animal in candidates 
                       if animal['features'].get(feature, 0) > 0.5)
        total = len(candidates)
        
        if yes_count == 0 or yes_count == total:
            return 0
        
        yes_ratio = yes_count / total
        no_ratio = 1 - yes_ratio
        
        entropy = -(yes_ratio * math.log2(yes_ratio) + no_ratio * math.log2(no_ratio))
        return entropy
    
    def get_best_question(self, candidates: List[Dict], asked_features: List[str]) -> Optional[Dict]:
        """Select optimal question using information gain"""
        available_questions = [q for q in self.questions 
                             if q['feature'] not in asked_features]
        
        if not available_questions:
            return None
        
        best_question = None
        max_gain = 0
        
        for question in available_questions:
            entropy = self.calculate_entropy(candidates, question['feature'])
            # Weight by question importance
            weighted_gain = entropy * question['weight']
            
            if weighted_gain > max_gain:
                max_gain = weighted_gain
                best_question = question
        
        return best_question
    
    def filter_candidates(self, answers: Dict[str, int]) -> List[Dict]:
        """Filter animals based on current answers"""
        candidates = []
        
        for animal in self.animals:
            matches = True
            for feature, answer in answers.items():
                animal_value = animal['features'].get(feature, 0)
                
                if answer == 1 and animal_value < 0.5:
                    matches = False
                    break
                elif answer == 0 and animal_value > 0.5:
                    matches = False
                    break
            
            if matches:
                candidates.append(animal)
        
        return candidates
    
    def make_guess(self, answers: Dict[str, int]) -> Tuple[Optional[Dict], float]:
        """Make best guess with confidence score"""
        candidates = self.filter_candidates(answers)
        
        if not candidates:
            return None, 0
        
        if len(candidates) == 1:
            return candidates[0], 0.95
        
        # Score candidates based on feature matches
        best_animal = None
        best_score = 0
        
        for animal in candidates:
            score = 0
            total_features = len(answers)
            
            for feature, answer in answers.items():
                animal_value = animal['features'].get(feature, 0)
                if (answer == 1 and animal_value > 0.5) or (answer == 0 and animal_value < 0.5):
                    score += 1
            
            normalized_score = score / total_features if total_features > 0 else 0
            
            if normalized_score > best_score:
                best_score = normalized_score
                best_animal = animal
        
        # Calculate confidence based on number of candidates and match quality
        confidence = min(0.95, max(0.1, best_score * (1 - (len(candidates) - 1) * 0.1)))
        
        return best_animal, confidence
    
    def learn_from_game(self, actual_animal: str, answers: Dict[str, int], description: str = ""):
        """Learn from incorrect guess"""
        # Find or create animal
        animal = None
        for a in self.animals:
            if a['name'].lower() == actual_animal.lower():
                animal = a
                break
        
        if not animal:
            animal = {"name": actual_animal, "features": {}}
            self.animals.append(animal)
        
        # Update features based on answers
        for feature, answer in answers.items():
            animal['features'][feature] = answer
        
        # Extract features from description
        if description:
            self.extract_features_from_description(animal, description)
        
        # Record game for analysis
        self.game_history.append({
            'date': datetime.now().isoformat(),
            'animal': actual_animal,
            'answers': answers.copy(),
            'description': description,
            'success': False
        })
        
        print(f"\n🧠 Learned about {actual_animal}! This will help me in future games.")
        self.save_data()
    
    def extract_features_from_description(self, animal: Dict, description: str):
        """Extract features from natural language description"""
        desc_lower = description.lower()
        
        feature_keywords = {
            'large': ['large', 'big', 'huge', 'massive', 'giant'],
            'small': ['small', 'tiny', 'little', 'miniature'],
            'aquatic': ['water', 'ocean', 'sea', 'swimming', 'aquatic'],
            'flies': ['fly', 'flying', 'wings', 'air', 'flight'],
            'domestic': ['pet', 'domestic', 'house', 'tame'],
            'wild': ['wild', 'jungle', 'forest', 'safari'],
            'carnivore': ['meat', 'carnivore', 'predator', 'hunter'],
            'herbivore': ['plants', 'grass', 'herbivore', 'vegetarian'],
            'fur': ['fur', 'furry', 'hairy'],
            'feathers': ['feather', 'feathered'],
            'scales': ['scale', 'scaled', 'scaly']
        }
        
        for feature, keywords in feature_keywords.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    animal['features'][feature] = 1
                    break
    
    def play_game(self):
        """Main game loop"""
        print("\n" + "="*50)
        print("🦁 Welcome to ML Animal Guesser!")
        print("="*50)
        print(f"Animals I know: {len(self.animals)}")
        print(f"Games played: {self.stats['played']}")
        if self.stats['played'] > 0:
            success_rate = (self.stats['correct'] / self.stats['played']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        print("\nThink of any animal and I'll try to guess it!")
        print("Answer with 'yes', 'y', 'no', or 'n'\n")
        
        answers = {}
        asked_features = []
        max_questions = 10
        
        for question_num in range(1, max_questions + 1):
            candidates = self.filter_candidates(answers)
            
            if len(candidates) <= 2 or question_num >= max_questions:
                break
            
            question = self.get_best_question(candidates, asked_features)
            if not question:
                break
            
            print(f"Question {question_num}: {question['text']}")
            
            while True:
                response = input("Your answer: ").lower().strip()
                if response in ['yes', 'y']:
                    answers[question['feature']] = 1
                    asked_features.append(question['feature'])
                    break
                elif response in ['no', 'n']:
                    answers[question['feature']] = 0
                    asked_features.append(question['feature'])
                    break
                else:
                    print("Please answer 'yes' or 'no'")
        
        # Make guess
        guess, confidence = self.make_guess(answers)
        
        if guess:
            print(f"\n🤔 I'm {confidence:.0%} confident...")
            print(f"Is your animal a {guess['name']}?")
            
            while True:
                response = input("Am I correct? (yes/no): ").lower().strip()
                if response in ['yes', 'y']:
                    self.stats['played'] += 1
                    self.stats['correct'] += 1
                    print("\n🎉 Yay! I guessed it correctly!")
                    self.save_data()
                    return
                elif response in ['no', 'n']:
                    break
                else:
                    print("Please answer 'yes' or 'no'")
        
        # Learning mode
        print("\n🧠 I didn't guess correctly. Help me learn!")
        actual_animal = input("What animal were you thinking of? ").strip()
        description = input("Can you describe it briefly? (optional): ").strip()
        
        self.stats['played'] += 1
        self.learn_from_game(actual_animal, answers, description)
    
    def show_stats(self):
        """Display learning statistics"""
        print("\n" + "="*30)
        print("📊 ML Learning Statistics")
        print("="*30)
        print(f"Total games played: {self.stats['played']}")
        print(f"Correct guesses: {self.stats['correct']}")
        if self.stats['played'] > 0:
            success_rate = (self.stats['correct'] / self.stats['played']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        print(f"Animals in database: {len(self.animals)}")
        print(f"Games recorded: {len(self.game_history)}")
        
        print("\nAnimals I know:")
        for i, animal in enumerate(self.animals, 1):
            features_count = len(animal['features'])
            print(f"{i:2d}. {animal['name']} ({features_count} features)")

def main():
    """Main application entry point"""
    ml_guesser = AnimalGuesserML()
    
    while True:
        print("\n" + "="*40)
        print("🦁 ML Animal Guesser")
        print("="*40)
        print("1. Play Game")
        print("2. View Statistics")
        print("3. Exit")
        
        choice = input("\nChoose an option (1-3): ").strip()
        
        if choice == '1':
            ml_guesser.play_game()
        elif choice == '2':
            ml_guesser.show_stats()
        elif choice == '3':
            print("\nThanks for playing! 🦁")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()