import random
import re
import string

# ============================================================
# KNOWLEDGE BASE
# ============================================================
KNOWLEDGE_BASE = [
    {
        'tag': 'greeting',
        'patterns': [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon',
            'good evening', 'howdy', 'sup', 'what is up', 'greetings'
        ],
        'responses': [
            'Hello! Welcome to CollegeBot. How can I help you today?',
            'Hi there! I am CollegeBot. Ask me anything about the college!',
            'Hey! Great to see you. What would you like to know?'
        ]
    },
    {
        'tag': 'farewell',
        'patterns': [
            'bye', 'goodbye', 'see you', 'see ya', 'take care',
            'quit', 'exit', 'later', 'good night', 'thanks bye'
        ],
        'responses': [
            'Goodbye! Best of luck with your studies!',
            'See you later! Feel free to come back anytime.',
            'Take care! Hope I was helpful today.'
        ]
    },
    {
        'tag': 'admission',
        'patterns': [
            'how to apply', 'admission process', 'how can i get admission',
            'what is the admission procedure', 'how do i join',
            'enrollment process', 'registration process', 'how to enroll'
        ],
        'responses': [
            'To apply, visit our official website and fill the online application form. '
            'You will need your 12th marksheet, ID proof, and passport photo.',
            'Admissions open every June. Complete the form at college-website.edu/apply '
            'and submit your documents before the deadline.'
        ]
    },
    {
        'tag': 'fees',
        'patterns': [
            'what are the fees', 'how much does it cost', 'fee structure',
            'tuition fees', 'course fee', 'how much is the fee',
            'college fees', 'semester fees', 'annual fees'
        ],
        'responses': [
            'Fee structure varies by course. Engineering: Rs 80,000/year. '
            'Commerce: Rs 40,000/year. Science: Rs 45,000/year. '
            'Scholarships available for merit students!'
        ]
    },
    {
        'tag': 'courses',
        'patterns': [
            'what courses are available', 'which programs do you offer',
            'list of courses', 'available programs', 'what can i study',
            'departments', 'what subjects', 'what degrees'
        ],
        'responses': [
            'We offer: B.Tech (CS, IT, ECE, Mech), BCA, BBA, B.Sc (Physics, Chemistry, Maths), '
            'B.Com, BA, MBA, MCA. Visit the Courses page for full details!'
        ]
    },
    {
        'tag': 'hostel',
        'patterns': [
            'hostel', 'accommodation', 'where to stay', 'is hostel available',
            'residential facility', 'do you have hostel', 'dormitory'
        ],
        'responses': [
            'Yes! Separate hostels are available for boys and girls. '
            'Monthly charges: Rs 6,000 (including meals). '
            'Contact the hostel warden at hostel@college.edu to book a room.'
        ]
    },
    {
        'tag': 'placements',
        'patterns': [
            'placement', 'job after college', 'campus recruitment',
            'highest package', 'placement record', 'companies that visit',
            'job opportunities', 'salary after graduation'
        ],
        'responses': [
            'Our placement rate is 92%! Top recruiters include Infosys, TCS, Wipro, '
            'Amazon, and Accenture. Highest package last year: Rs 18 LPA. '
            'Average package: Rs 6.5 LPA. Our placement cell starts prep from 3rd year.'
        ]
    },
    {
        'tag': 'thanks',
        'patterns': [
            'thank you', 'thanks', 'thank you so much', 'thanks a lot',
            'that was helpful', 'great', 'awesome', 'perfect', 'wonderful'
        ],
        'responses': [
            'You are welcome! Is there anything else I can help with?',
            'Happy to help! Feel free to ask more questions.',
            'Glad I could assist! What else would you like to know?'
        ]
    },
    {
        'tag': 'unknown',
        'patterns': [],
        'responses': [
            'I am not sure I understand. Could you rephrase that?',
            'Hmm, I do not have information on that. Try asking about admissions, fees, or courses!',
            'I did not quite get that. I can help with admissions, fees, hostel, placements, and courses!'
        ]
    }
]


# ============================================================
# PART 2: TEXT PREPROCESSING
# ============================================================
def preprocess(text):
    """
    Clean and normalise user input.
    - Convert to lowercase
    - Remove punctuation
    - Remove extra spaces
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = ' '.join(text.split())
    return text


# ============================================================
# PART 3: KEYWORD OVERLAP MATCHING ENGINE
# ============================================================
def find_best_match(user_input):
    """
    Compare cleaned user input against all patterns.
    Returns: (tag, response_string)
    """
    cleaned = preprocess(user_input)
    user_words = set(cleaned.split())

    best_tag = 'unknown'
    best_score = 0

    for entry in KNOWLEDGE_BASE:
        if entry['tag'] == 'unknown':
            continue
        for pattern in entry['patterns']:
            clean_pattern = preprocess(pattern)
            pattern_words = set(clean_pattern.split())
            overlap = len(user_words & pattern_words)
            if overlap > best_score:
                best_score = overlap
                best_tag = entry['tag']

    for entry in KNOWLEDGE_BASE:
        if entry['tag'] == best_tag:
            return best_tag, random.choice(entry['responses'])

    for entry in KNOWLEDGE_BASE:
        if entry['tag'] == 'unknown':
            return 'unknown', random.choice(entry['responses'])


# ============================================================
# PART 5: REGEX-BASED MATCHING (tried first, falls back to Part 3)
# ============================================================
REGEX_PATTERNS = [
    (r'\b(hi|hello|hey|howdy|greetings)\b', 'greeting'),
    (r'\b(bye|goodbye|see\s+you|take\s+care|exit|quit)\b', 'farewell'),
    (r'\b(fee|fees|cost|price|charges|tuition)\b', 'fees'),
    (r'\b(admission|apply|enroll|register|join)\b', 'admission'),
    (r'\b(course|courses|program|degree|subject|department)\b', 'courses'),
    (r'\b(hostel|accommodation|stay|room|dormitory)\b', 'hostel'),
    (r'\b(placement|job|recruit|salary|package|career)\b', 'placements'),
    (r'\b(thank|thanks|great|awesome|helpful|perfect)\b', 'thanks'),
]


def regex_match(user_input):
    """Try regex matching before falling back to keyword matching."""
    cleaned = preprocess(user_input)
    for pattern, tag in REGEX_PATTERNS:
        if re.search(pattern, cleaned):
            for entry in KNOWLEDGE_BASE:
                if entry['tag'] == tag:
                    return tag, random.choice(entry['responses'])
    return find_best_match(user_input)


# ============================================================
# PART 4: MAIN CONVERSATION LOOP
# ============================================================
def chat():
    """Main chatbot conversation loop."""
    print('=' * 55)
    print(' CollegeBot - Your AI College Assistant')
    print(' Powered by Main Flow Services and Technologies')
    print('=' * 55)
    print('Type your message and press Enter. Type "bye" to exit.\n')

    conversation_log = []

    while True:
        user_input = input('You: ').strip()

        if not user_input:
            print('Bot: Please type something!\n')
            continue

        conversation_log.append({'role': 'user', 'message': user_input})

        tag, response = regex_match(user_input)

        print(f'Bot: {response}\n')
        conversation_log.append({'role': 'bot', 'message': response, 'tag': tag})

        if tag == 'farewell':
            print(f'[Session ended. Total exchanges: {len(conversation_log)//2}]')
            break


if __name__ == '__main__':
    chat()