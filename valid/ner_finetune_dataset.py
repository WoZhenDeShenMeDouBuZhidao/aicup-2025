import random
import collections

NER_TAG_MEANING = {
    'PATIENT': """Refers to a patient's name (full, first, last, with title; e.g., "Johnathan Doe", "Mary", "Mr. Smith", "王大明", "麗華", "陳先生").""",
    'DOCTOR': """Refers to a doctor's name (e.g., "Dr. Emily White", "Dr. Jones", "Chang", "王醫師", "張志明醫師"); often prefixed with "Dr." or a title like "醫師".""",
    'FAMILYNAME': """Refers to a person's surname in a non-patient/doctor context or as a general family reference (e.g., "Williams", "Peterson family", "Chen's", "張", "黃氏").""",
    'PERSONALNAME': """Refers to a person's given or full name not tagged as PATIENT/DOCTOR (e.g., "Michael", "Sarah", "James Anderson", "家豪", "小美", "陳建明").""",
    'PROFESSION':"""Refers to a person's job or profession (e.g., "journalist", "president", "selling soap", "經理", "工程師", "律師", "主教"); includes job titles or descriptive phrases.""",
    'ROOM': """Specific room identifiers within a medical building (e.g., "Room 402", "Ward 3 East", "ICU Bed 5", "402號房", "東三區病房"); alphanumeric codes or names associated with rooms/beds.""",
    'DEPARTMENT': """Specific medical department within an institution (e.g., "Cardiology Department", "Pathology", "Oncology Ward", "心臟科", "放射科"); often includes "Department", "Unit", "Ward", "Area", "科", "部", "中心".""",
    'HOSPITAL': """Name of a hospital or major medical center (e.g., "Mercy General Hospital", "St. Luke's Medical Center", "仁愛綜合醫院", "聖路加醫學中心"); often includes "Hospital", "Medical Center", "Healthcare", "醫院", "醫學中心".""",
    'ORGANIZATION': """Company, non-medical institution, or other organized body (e.g., "Costco Wholesale Corporation", "Tech Solutions Inc.", "Stanford University", "China Airlines", "國泰人壽", "宏碁公司"); includes names of businesses, foundations, universities not tagged as hospital/department.""",
    'STREET': """Street names and numbers (e.g., "123 Willow Creek Drive", "Main St", "中山北路一段25號", "忠孝東路"); includes number, name, and type (Road, Ave, St, 路, 街, 巷, 弄, 號).""",
    'CITY': """Names of cities or towns (e.g., "Springfield", "Denver", "London", "台北市", "高雄"); capitalized names, often followed by state/country.""",
    'DISTRICT': """Named sub-regions of a city or larger administrative area (e.g., "Bronx", "Westminster borough", "大安區", "板橋區"); may include "District", "Borough", "區".""",
    'COUNTY': """Named administrative division (e.g., "Dade County", "Shire of Broome", "宜蘭縣", "屏東縣"); often includes "County", "Shire", "縣".""",
    'STATE': """Major administrative division of a country (e.g., "Texas", "NY", "Victoria", "SA", "加州", "台灣省"); full names or abbreviations, "州", "省".""",
    'COUNTRY': """Names of sovereign nations (e.g., "Canada", "United Kingdom", "USA", "日本", "英國", "美國"); full names or common unambiguous abbreviations.""",
    'ZIP': """Postal codes for mail (e.g., "9010", "2067", "110", "30078"); 4-digit or 5-digit numeric sequences.""",
    'LOCATION-OTHER': """Geographical entities not fitting other categories (e.g., "Midwest", "Silicon Valley", "Grand Canyon", "南部地區", "陽明山國家公園"); named regions, landmarks, imprecise locations.""",
    'AGE': """Numerical representation of a person's age (e.g., "35-year-old", "10", "age 62", "40s", "37歲", "十歲"); numbers often with "years old", "y.o.", "歲", or stand with a person's name in context.""",
    'DATE': """Specific dates, months, years, days of the week, relative dates (e.g., "June 15, 2024", "03/05/2023", "now", "next Monday", "August 1988", "2024年6月15日", "民國112年3月5日", "昨天", "上週"); various formats and textual representations.""",
    'TIME': """Specific times of day or relative times (e.g., "8:00 AM", "14:30", "this morning", "noon", "last night around 10 p.m.", "上午8點", "14時30分", "今天早上", "中午"); clock times (HH:MM, AM/PM) and textual phrases.""",
    'DURATION': """Spans of time (e.g., "three weeks", "24 hours", "several years", "a few days", "past few years", "a long time", "a while", "whole weekend", "三個星期", "這幾天", "春天", "這周期間"); numbers with time units or phrases.""",
    'SET': """Recurring events or frequencies (e.g., "twice a day", "every day", "per month", "weekly", "every 15 minutes", "每日服藥兩次", "每隔一天", "每月一次"); describes how often an event occurs.""",
    'MEDICAL_RECORD_NUMBER': """Unique identifier for a patient's medical record (e.g., "405974.QBV", "7890123.MRN", "2805065.FMV"); alphanumeric, most of them satisfy the format {digit sequence}.{alphabet sequence}""",
    'ID_NUMBER': """Other identification numbers (lab, specimen, national ID, insurance, etc.) (e.g., "73C10671", "lab number 92M63178", "NHI ZYX987A", "Accession ID: L2023-5678", "身分證字號為A123456789", "檢體編號 S2023-001"); alphanumeric, often context-specific.""",
}

def load_transcriptions(filepath):
    """Loads transcriptions from task1_answer.txt."""
    transcriptions = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    data_id, transcript = parts
                    transcriptions[data_id] = transcript
                else:
                    print(f"Warning: Skipping malformed line in transcriptions file: {line}")
    except FileNotFoundError:
        print(f"Error: Transcription file not found at {filepath}")
    return transcriptions

def load_ner_labels(filepath):
    """Loads NER labels from task2_answer.txt."""
    ner_labels = collections.defaultdict(list)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 5:
                    data_id, ner_type, start_time_str, end_time_str, ner_content = parts
                    try:
                        start_time = float(start_time_str)
                        end_time = float(end_time_str)
                        ner_labels[data_id].append({
                            "type": ner_type,
                            "start_time": start_time,
                            "end_time": end_time,
                            "content": ner_content
                        })
                    except ValueError:
                        print(f"Warning: Skipping malformed NER entry (time conversion error): {line}")
                else:
                    print(f"Warning: Skipping malformed line in NER labels file: {line}")
    except FileNotFoundError:
        print(f"Error: NER labels file not found at {filepath}")
    return ner_labels

def format_single_transcript(data_id, transcript, ner_entries_for_id, specified_ner_types):
    """
    Formats a single transcript by adding NER tags.

    Returns:
        Tuple[str, bool]: (assistant_prompt_string, was_formatted_boolean)
    """
    relevant_ners = []
    for ner in ner_entries_for_id:
        if ner["type"] in specified_ner_types:
            relevant_ners.append(ner)

    # Sort NERs by start_time to process them in their spoken/natural order
    # This helps in correctly finding character offsets sequentially
    relevant_ners.sort(key=lambda x: x["start_time"])

    # Find character offsets for each relevant NER in the original transcript
    char_offset_ners = []
    current_search_offset = 0
    for ner in relevant_ners:
        ner_content = ner["content"]
        try:
            # Find the ner_content starting from the current_search_offset
            start_char = transcript.find(ner_content, current_search_offset)
            if start_char != -1:
                end_char = start_char + len(ner_content)
                char_offset_ners.append({
                    "start_char": start_char,
                    "end_char": end_char,
                    "type": ner["type"],
                    "content": ner_content # Original content from transcript slice
                })
                # Advance search offset to find subsequent occurrences correctly
                current_search_offset = start_char + 1 # Allow finding adjacent or overlapping entities if sorted by time
            else:
                # This specific NER content instance (expected at this time) was not found
                # print(f"Warning (ID: {data_id}): NER content '{ner_content}' (type: {ner['type']}) expected around time {ner['start_time']} not found sequentially in transcript after offset {current_search_offset}.")
                pass # Optionally log or handle. For now, we skip if not found sequentially.

        except Exception as e:
            print(f"Error finding NER content '{ner_content}' in transcript ID {data_id}: {e}")


    if not char_offset_ners:
        return transcript, False # No relevant NERs were found and tagged

    # Sort entities by their start character in descending order for tagging
    # This ensures that inserting tags from right to left doesn't mess up indices
    # of entities appearing earlier in the string.
    char_offset_ners.sort(key=lambda x: x["start_char"], reverse=True)

    assistant_prompt = transcript
    for ner_info in char_offset_ners:
        s = ner_info["start_char"]
        e = ner_info["end_char"]
        ner_type = ner_info["type"]
        # Grab the actual slice from the current assistant_prompt state to allow nesting
        original_slice = assistant_prompt[s:e]

        tag_open = f"<{ner_type}>"
        tag_close = f"</{ner_type}>"
        
        assistant_prompt = assistant_prompt[:s] + tag_open + original_slice + tag_close + assistant_prompt[e:]

    return assistant_prompt, True

def generate_ner_training_data(task1_filepath, task2_filepath, specified_ner_types_list):
    """
    Generates NER training data based on transcriptions and NER labels.
    """
    transcriptions_map = load_transcriptions(task1_filepath)
    all_ner_labels_map = load_ner_labels(task2_filepath)
    ner_types = ""
    for ner_type in specified_ner_types_list:
        ner_types += f"- {ner_type}: {NER_TAG_MEANING[ner_type]}\n"

    if not transcriptions_map:
        print("No transcriptions loaded. Exiting.")
        return []

    SYSTEM_PROMPT = f"""You are an expert in Named Entity Recognition.
Your task is to identify and tag entities in the provided Electronic Health Records.
If there is no specified entity, output original content.
Entity Type you need to recognize:
{ner_types}"""

    all_processed_items = []

    for data_id, transcript_text in transcriptions_map.items():
        user_prompt_text = transcript_text
        ner_entries = all_ner_labels_map.get(data_id, [])

        assistant_prompt_text, item_is_formatted = format_single_transcript(
            data_id,
            transcript_text,
            ner_entries,
            specified_ner_types_list
        )

        all_processed_items.append({
            "data_id": data_id, # Keep for reference, can be removed from final dict
            "system": SYSTEM_PROMPT,
            "user": user_prompt_text,
            "assistant": assistant_prompt_text,
            "is_formatted": item_is_formatted
        })

    formatted_outputs = [item for item in all_processed_items if item["is_formatted"]]
    unformatted_outputs = [item for item in all_processed_items if not item["is_formatted"]]

    num_formatted = len(formatted_outputs)
    num_unformatted_initial = len(unformatted_outputs)

    # We want: num_formatted > num_unformatted_kept
    # So, num_unformatted_kept can be at most num_formatted // 2
    num_unformatted_to_keep = max(0, num_formatted // 2)

    if num_unformatted_initial > num_unformatted_to_keep:
        # We have more unformatted items than we are allowed to keep, so sample down.
        unformatted_outputs_final = random.sample(unformatted_outputs, num_unformatted_to_keep)
    else:
        # We have fewer or equal unformatted items than allowed, so keep all of them.
        unformatted_outputs_final = unformatted_outputs
    
    # Prepare final list without temporary keys
    final_training_data = []
    for item in formatted_outputs + unformatted_outputs_final:
        final_training_data.append({
            "system": item["system"],
            "user": item["user"],
            "assistant": item["assistant"]
        })

    random.shuffle(final_training_data) # Good practice to shuffle the combined dataset

    print(f"Generated {len(final_training_data)} training examples.")
    print(f"  Formatted examples: {num_formatted}")
    print(f"  Unformatted examples initially: {num_unformatted_initial}")
    print(f"  Unformatted examples kept: {len(unformatted_outputs_final)}")

    return final_training_data

if __name__ == '__main__':
    # --- Configuration ---
    task1_file = 'task1_answer.txt' # Path to your ASR transcriptions
    task2_file = 'task2_answer.txt' # Path to your NER labels

    # Define the NER types you want to tag in this run
    # Example: Process only 'DATE' and 'TIME' entities
    # specified_types_for_this_run = ['DATE', 'TIME']
    # Example from your files:
    # specified_types_for_this_run = ['PATIENT', 'DOCTOR', 'FAMILYNAME', 'PERSONALNAME', 'PROFESSION']
    # specified_types_for_this_run = ['ROOM', 'DEPARTMENT', 'HOSPITAL', 'ORGANIZATION', 'STREET', 'CITY', 'DISTRICT', 'COUNTY', 'STATE', 'COUNTRY', 'ZIP', 'LOCATION-OTHER']
    # specified_types_for_this_run = ['AGE', 'DATE', 'TIME', 'DURATION', 'SET']
    # specified_types_for_this_run = ['MEDICAL_RECORD_NUMBER', 'ID_NUMBER']

    # specified_types_for_this_run = ['PATIENT', 'DOCTOR', 'FAMILYNAME', 'PERSONALNAME']
    # specified_types_for_this_run = ['PROFESSION']
    specified_types_for_this_run = ['ROOM', 'DEPARTMENT', 'HOSPITAL', 'ORGANIZATION']
    # specified_types_for_this_run = ['STREET', 'CITY', 'DISTRICT', 'COUNTY', 'STATE', 'COUNTRY', 'ZIP', 'LOCATION-OTHER']
    # specified_types_for_this_run = ['AGE', 'DATE', 'TIME', 'DURATION', 'SET']
    # specified_types_for_this_run = ['MEDICAL_RECORD_NUMBER', 'ID_NUMBER']


    # --- Generate Data ---
    training_data = generate_ner_training_data(task1_file, task2_file, specified_types_for_this_run)

    # --- Output/Save Data ---
    # You can now save this 'training_data' list to a file (e.g., JSONL)
    # or use it directly. For demonstration, printing the first few:
    print("\n--- Sample Generated Data ---")
    for i, example in enumerate(training_data[:3]): # Print first 3 examples
        print(f"\nExample {i+1}:")
        print(f"  System: {example['system']}")
        print(f"  User: {example['user']}")
        print(f"  Assistant: {example['assistant']}")

    # Example of saving to a JSONL file:
    import json
    output_filename = f"ner_finetune_trainset_{'_'.join(specified_types_for_this_run)}.jsonl"
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for entry in training_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')
    print(f"\nSaved training data to {output_filename}")