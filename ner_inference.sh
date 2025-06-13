PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4 python ner_inference.py --ner_types 'PATIENT' 'DOCTOR' 'FAMILYNAME' 'PERSONALNAME' \
    --adapter_step 'checkpoint-765' \
    --user1 'The patient, Ms. Jane Doe, was assessed by Dr. Smith. her son John mentioned ⁇ the Doe family history of allergies.' \
    --asit1 'The patient, <PATIENT>Ms. Jane Doe</PATIENT>, was assessed by <DOCTOR>Dr. Smith</DOCTOR>. her son <PERSONALNAME>John</PERSONALNAME> mentioned ⁇ the <FAMILYNAME>Doe</FAMILYNAME> family history of allergies.' \
    --user2 'Dr. Chang reviewed Robert Miller ⁇ s chart. Anna, his wife née Wilson, is his emergency contact' \
    --asit2 'Dr. <DOCTOR>Chang</DOCTOR> reviewed <PATIENT>Robert Miller</PATIENT> ⁇ s chart. <PERSONALNAME>Anna</PERSONALNAME>, his wife née <FAMILYNAME>Wilson</FAMILYNAME>, is his emergency contact' \
    --user3 '病患陳心怡女士由王醫師評估其女兒小美提及陳氏家族的過敏史' \
    --asit3 '病患<PATIENT>陳心怡</PATIENT>女士由<DOCTOR>王</DOCTOR>醫師評估其女兒<PERSONALNAME>小美</PERSONALNAME>提及<FAMILYNAME>陳</FAMILYNAME>氏家族的過敏史' \
    --user4 '... regarding patient David Green, caregiver Sarah discussed with Dr. Peterson the Green family ⁇ s medical background. ⁇' \
    --asit4 '... regarding patient <PATIENT>David Green</PATIENT>, caregiver <PERSONALNAME>Sarah</PERSONALNAME> discussed with Dr. <DOCTOR>Peterson</DOCTOR> the <FAMILYNAME>Green</FAMILYNAME> family ⁇ s medical background. ⁇' \
    --user5 '林醫師檢閱了病患李大文的病歷其看護張先生記錄了李家的聯絡方式' \
    --asit5 '<DOCTOR>林</DOCTOR>醫師檢閱了病患<PATIENT>李大文</PATIENT>的病歷其看護<PERSONALNAME>張先生</PERSONALNAME>記錄了<FAMILYNAME>李</FAMILYNAME>家的聯絡方式' \
    > PATIENT.txt

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4 python ner_inference.py --ner_types 'PROFESSION' \
    --adapter_step 'checkpoint-72' \
    --user1 'Patient states his spouse is a... software engineer.' \
    --asit1 'Patient states his spouse is a... <PROFESSION>software engineer</PROFESSION>.' \
    --user2 'her previous work as a teacher involved long hours standing ' \
    --asit2 'her previous work as a <PROFESSION>teacher</PROFESSION> involved long hours standing ' \
    --user3 '病人表示其配偶是一名藝術家' \
    --asit3 '病人表示其配偶是一名<PROFESSION>藝術家</PROFESSION>' \
    --user4 'The consultant, a renowned cardiologist,  ⁇ reviewed the case.' \
    --asit4 'The <PROFESSION>consultant</PROFESSION>, a renowned <PROFESSION>cardiologist</PROFESSION>,  ⁇ reviewed the case.' \
    --user5 '他作為一名創作者工作壓力較大' \
    --asit5 '他作為一名<PROFESSION>創作者</PROFESSION>工作壓力較大' \
    > PROFESSION.txt

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4 python ner_inference.py --ner_types 'ROOM' 'DEPARTMENT' 'HOSPITAL' 'ORGANIZATION' \
    --adapter_step 'checkpoint-76' \
    --user1 'Patient admitted to Room 301 of the Cardiology Department at Mayo Hospital on 123 Main St., Rochester, Minnesota, ZIP 55905, United States.' \
    --asit1 'Patient admitted to <ROOM>Room 301</ROOM> of the <DEPARTMENT>Cardiology Department</DEPARTMENT> at <HOSPITAL>Mayo Hospital</HOSPITAL> on 123 Main St., Rochester, Minnesota, 55905, United States.' \
    --user2 'Referral sent by the Red Cross Society located in the Financial District of New York County, New York.' \
    --asit2 'Referral sent by the <ORGANIZATION>Red Cross Society</ORGANIZATION> located in the Financial District of New York County, New York, New York.' \
    --user3 '病人轉送到台北榮民總醫院內科部的5樓501病房地址位於台北市信義區忠孝東路五段100號郵遞區號110' \
    --asit3 '病人轉送到<HOSPITAL>台北榮民總醫院</HOSPITAL><DEPARTMENT>內科部</DEPARTMENT>的<ROOM>5樓501病房</ROOM>地址位於台北市信義區忠孝東路五段100號郵遞區號110' \
    --user4 'The mobile clinic from the World Health Organization set up <LOCATION-OTHER>near Central Park in Manhattan, New York, New York.' \
    --asit4 'The mobile clinic from the <ORGANIZATION>World Health Organization</ORGANIZATION> set up near Central Park in Manhattan, New York, New York.' \
    --user5 '慈濟基金會在花蓮縣吉安鄉設立了流動醫療站靠近美崙溪畔方便偏鄉居民就診' \
    --asit5 '<ORGANIZATION>慈濟基金會</ORGANIZATION>在花蓮縣吉安鄉設立了流動醫療站靠近美崙溪畔方便偏鄉居民就診' \
    > ROOM.txt

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4 python ner_inference.py --ner_types 'STREET' 'CITY' 'DISTRICT' 'COUNTY' 'STATE' 'COUNTRY' 'ZIP' 'LOCATION-OTHER' \
    --adapter_step 'checkpoint-368' \
    --user1 'Patient resides at 123 Oak Street, Anytown ⁇ North District, Lincoln County, CA 90210, USA, near the Downtown Plaza.' \
    --asit1 'Patient resides at <STREET>123 Oak Street</STREET>, <CITY>Anytown</CITY> ⁇ <DISTRICT>North District</DISTRICT>, <COUNTY>Lincoln</COUNTY> County, <STATE>CA</STATE> <ZIP>90210</ZIP>, <COUNTRY>USA</COUNTRY>, near the <LOCATION-OTHER>Downtown Plaza</LOCATION-OTHER>.' \
    --user2 '...previous address: 45 Pine Avenue, Springfield, West End district, Dane County, IL 62701, United States, located in the River Valley area.' \
    --asit2 '...previous address: <STREET>45 Pine Avenue</STREET>, <CITY>Springfield</CITY>, <DISTRICT>West End district</DISTRICT>, <COUNTY>Dane</COUNTY> County, <STATE>IL</STATE> <ZIP>62701</ZIP>, <COUNTRY>United States</COUNTRY>, located in the <LOCATION-OTHER>River Valley area</LOCATION-OTHER>.' \
    --user3 '病患地址為中華民國台灣省桃園市中壢區中山路100號郵遞區號320靠近中央大學' \
    --asit3 '病患地址為<COUNTRY>中華民國</COUNTRY><STATE>台灣省</STATE><CITY>桃園市</CITY><DISTRICT>中壢區</DISTRICT><STREET>中山路100號</STREET>郵遞區號<ZIP>320</ZIP>靠近<LOCATION-OTHER>中央大學</LOCATION-OTHER>' \
    --user4 'he traveled from his home on George Street, CBD, Sydney, Yorkshire, NSW 2000, Australia, which is close to the Harbourfront.' \
    --asit4 'he traveled from his home on <STREET>George Street</STREET>, <DISTRICT>CBD</DISTRICT>, <CITY>Sydney</CITY>, <COUNTY>Yorkshire</COUNTY>, <STATE>NSW</STATE> <ZIP>2000</ZIP>, <COUNTRY>Australia</COUNTRY>, which is close to the <LOCATION-OTHER>Harbourfront</LOCATION-OTHER>.' \
    --user5 '檔案記載其先前居住於美國加州洛杉磯縣洛杉磯市好萊塢區日落大道789號郵編90028位於天使之城' \
    --asit5 '檔案記載其先前居住於<COUNTRY>美國</COUNTRY><STATE>加州</STATE><COUNTY>洛杉磯縣</COUNTY><CITY>洛杉磯市</CITY><DISTRICT>好萊塢區</DISTRICT><STREET>日落大道789號</STREET>郵編<ZIP>90028</ZIP>位於<LOCATION-OTHER>天使之城</LOCATION-OTHER>' \
    > STREET.txt

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4 python ner_inference.py --ner_types 'AGE' 'DATE' 'TIME' 'DURATION' 'SET' \
    --adapter_step 'checkpoint-832' \
    --user1 'patient is a 65-year-old male presenting with chest pain. He reports symptoms since June 10, 2025 at four o clock. He has been on medication for 3 weeks and takes aspirin twice daily.' \
    --asit1 'patient is a <AGE>65</AGE>-year-old male presenting with chest pain. He reports symptoms since <DATE>June 10, 2025</DATE> at <TIME>four</TIME> o clock. He has been on medication for <DURATION>3 weeks</DURATION> and takes aspirin <SET>twice daily</SET>.' \
    --user2 'On May 1, 2025, in the morning, a fifty-year-old female underwent MRI. She rested for ... two hours before the exam.'\
    --asit2 'On <DATE>May 1, 2025</DATE>, in the <TIME>morning</TIME>, a <AGE>45</AGE>-year-old female underwent MRI. She rested for ... <DURATION>two hours</DURATION> before the exam.' \
    --user3 '兒童8歲於2024年12月20日晚上20:30意外跌倒需觀察每四小時檢查一次預計48小時後出院' \
    --asit3 '兒童<AGE>8歲</AGE>於<DATE>2024年12月20日</DATE>晚上<TIME>20:30</TIME>意外跌倒需觀察<SET>每四小時檢查一次</SET>預計<DURATION>48小時</DURATION>後出院' \
    --user4 'Medication schedule: 500 mg amoxicillin at today ⁇ s eight and 19:00, to be repeated every 8 hours. The patient is 30 and he started therapy on July 15, 2024 and will continue for six months.' \
    --asit4 'Medication schedule: 500 mg amoxicillin at today ⁇ s <TIME>eight</TIME> and <TIME>19:00</TIME>, to be repeated <SET>every 8 hours</SET>. The patient is a <AGE>30</AGE> and he started therapy on <DATE>July 15, 2024</DATE> and will continue for <DURATION>six months</DURATION>.' \
    --user5 '患者為50歲男性於2025年6月1日上午09:00到院主訴咳嗽持續兩周服用止咳藥物每天三次' \
    --asit5 '患者為<AGE>50歲</AGE>男性於<DATE>2025年6月1日</DATE>上午<TIME>09:00</TIME>到院主訴咳嗽持續<DURATION>兩周</DURATION>服用止咳藥物<SET>每天三次</SET>' \
    > AGE.txt

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4 python ner_inference.py --ner_types 'MEDICAL_RECORD_NUMBER' 'ID_NUMBER' \
    --adapter_step 'checkpoint-217' \
    --user1 'The patient’s chart indicates medical record 93A5621.PQR with a follow-up ⁇ scheduled next week.' \
    --asit1 'The patient’s chart indicates medical record <MEDICAL_RECORD_NUMBER>93A5621.PQR</MEDICAL_RECORD_NUMBER> with a follow-up ⁇ scheduled next week.' \
    --user2 'Lab results were filed under ID number ... 55Z123789 and assigned to Dr. Smith.' \
    --asit2 'Lab results were filed under ID number ... <ID_NUMBER>55Z123789</ID_NUMBER> and assigned to Dr. Smith.' \
    --user3 '患者張先生之病歷號為7802345.XYZ門診序號23B55400請協助查詢相關資訊' \
    --asit3 '患者張先生之病歷號為<MEDICAL_RECORD_NUMBER>7802345.XYZ</MEDICAL_RECORD_NUMBER>門診序號<ID_NUMBER>23B55400</ID_NUMBER>請協助查詢相關資訊' \
    --user4 ' the medical report for Jenny Lopez references MRN 1209874.ABC and lab code 89M76231.' \
    --asit4 ' the medical report for Jenny Lopez references MRN <MEDICAL_RECORD_NUMBER>1209874.ABC</MEDICAL_RECORD_NUMBER> and lab code <ID_NUMBER>89M76231</ID_NUMBER>.' \
    --user5 '此病例檔案中住院編號4521876.DEF與身分證號A123456789尚未更新' \
    --asit5 '此病例檔案中住院編號<MEDICAL_RECORD_NUMBER>4521876.DEF</MEDICAL_RECORD_NUMBER>與身分證號<ID_NUMBER>A123456789</ID_NUMBER>尚未更新' \
    > MEDICAL_RECORD_NUMBER.txt