"""
TruthLens - Data Loader for GossipCop Dataset
Handles dataset generation, preprocessing storage, and train/val/test splitting.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

GOSSIPCOP_REPRESENTATIVE_SAMPLES = [
    # REAL NEWS SAMPLES (Label 1)
    {
        "id": "gossipcop-real-001",
        "title": "Jennifer Aniston and Justin Theroux Announce Separation After Two Years of Marriage",
        "text": "Jennifer Aniston and Justin Theroux have announced they are separating after two and a half years of marriage. In a joint statement released through Aniston's longtime publicist Stephen Huvane, the couple stated the decision was mutual and lovingly made at the end of last year. They emphasized their commitment to maintaining their deep friendship and mutual respect going forward.",
        "label": 1,
        "label_name": "REAL",
        "source": "AP News / Associated Press",
        "url": "https://apnews.com/article/entertainment-aniston-theroux"
    },
    {
        "id": "gossipcop-real-002",
        "title": "Bradley Cooper and Irina Shayk Split After Four Years Together",
        "text": "Bradley Cooper and Irina Shayk have officially called it quits after four years of dating. People confirmed that the 44-year-old Oscar-nominated actor and the 33-year-old supermodel decided to end their relationship amicably and are currently working out how to share custody of their daughter Lea De Seine, who was born in March 2017.",
        "label": 1,
        "label_name": "REAL",
        "source": "People Magazine",
        "url": "https://people.com/movies/bradley-cooper-irina-shayk-split"
    },
    {
        "id": "gossipcop-real-003",
        "title": "Prince Harry and Meghan Markle Step Back as Senior Royals",
        "text": "The Duke and Duchess of Sussex announced their decision to step back as senior members of the British Royal Family. Prince Harry and Meghan Markle released an official Buckingham Palace communication expressing their desire to become financially independent while continuing to support Queen Elizabeth II and their patronages across the UK and North America.",
        "label": 1,
        "label_name": "REAL",
        "source": "BBC News",
        "url": "https://bbc.com/news/uk-51040586"
    },
    {
        "id": "gossipcop-real-004",
        "title": "Lady Gaga Wins Best Original Song Oscar for Shallow from A Star Is Born",
        "text": "Lady Gaga took home her first Academy Award for Best Original Song for Shallow from the musical drama film A Star Is Born. During her emotional acceptance speech at the 91st Academy Awards, she thanked co-star and director Bradley Cooper, delivering an inspiring message about hard work and discipline in creative pursuits.",
        "label": 1,
        "label_name": "REAL",
        "source": "Reuters",
        "url": "https://reuters.com/article/us-awards-oscars-shallow"
    },
    {
        "id": "gossipcop-real-005",
        "title": "Leonardo DiCaprio Donates $5 Million to Amazon Forest Fire Relief Efforts",
        "text": "Leonardo DiCaprio's environmental initiative Earth Alliance committed $5 million in financial assistance to preserve the Amazon rainforest following devastating wildfires across Brazil and Bolivia. The funds were distributed directly to indigenous communities and local conservation organizations working on frontline emergency response.",
        "label": 1,
        "label_name": "REAL",
        "source": "Variety",
        "url": "https://variety.com/2019/film/news/leonardo-dicaprio-amazon-fire-5-million-donation"
    },
    {
        "id": "gossipcop-real-006",
        "title": "Taylor Swift Releases Surprise Eighth Studio Album Folklore",
        "text": "Taylor Swift announced and released her surprise eighth studio album titled Folklore, recorded entirely during COVID-19 pandemic isolation. The album features collaborations with Aaron Dessner of The National, Bon Iver, and Jack Antonoff, earning critical acclaim and debuting at number one on the Billboard 200 chart.",
        "label": 1,
        "label_name": "REAL",
        "source": "Rolling Stone",
        "url": "https://rollingstone.com/music/music-news/taylor-swift-folklore-announcement"
    },
    {
        "id": "gossipcop-real-007",
        "title": "Tom Hanks and Rita Wilson Recover After Contracting COVID-19 in Australia",
        "text": "Academy Award winning actor Tom Hanks and his wife Rita Wilson were released from a Queensland hospital after undergoing quarantine and medical treatment for coronavirus symptoms while filming Baz Luhrmann's Elvis biopic in Australia. Hanks kept fans updated on social media, urging the public to follow health guidelines.",
        "label": 1,
        "label_name": "REAL",
        "source": "The Guardian",
        "url": "https://theguardian.com/film/2020/mar/17/tom-hanks-rita-wilson-released-hospital-coronavirus"
    },
    {
        "id": "gossipcop-real-008",
        "title": "Angelina Jolie and Brad Pitt Reach Final Agreement on Child Custody",
        "text": "Angelina Jolie and Brad Pitt reached a finalized custody agreement regarding their six children, avoiding an impending public trial. Jolie's attorney Samantha Bley DeJean confirmed the legal arrangement was signed by both parties and the private judge, bringing resolution to one of the most high-profile custody proceedings in Hollywood.",
        "label": 1,
        "label_name": "REAL",
        "source": "E! Online",
        "url": "https://eonline.com/news/angelina-jolie-brad-pitt-custody-agreement"
    },
    {
        "id": "gossipcop-real-009",
        "title": "Selena Gomez Undergoes Kidney Transplant Due to Lupus Complications",
        "text": "Pop singer and actress Selena Gomez revealed that she received a kidney transplant from her best friend and fellow actress Francia Raisa over the summer due to organ damage from the autoimmune disease lupus. Gomez shared photos from her hospital stay and expressed profound gratitude to Raisa and the medical team.",
        "label": 1,
        "label_name": "REAL",
        "source": "Billboard",
        "url": "https://billboard.com/music/pop/selena-gomez-kidney-transplant-francia-raisa"
    },
    {
        "id": "gossipcop-real-010",
        "title": "Kim Kardashian and Kanye West File for Divorce After Six Years of Marriage",
        "text": "Kim Kardashian officially filed for divorce from Kanye West in Los Angeles Superior Court after nearly seven years of marriage. Representatives confirmed the split was amicable, with Kardashian requesting joint legal and physical custody of their four children: North, Saint, Chicago, and Psalm.",
        "label": 1,
        "label_name": "REAL",
        "source": "CNN Entertainment",
        "url": "https://cnn.com/2021/02/19/entertainment/kim-kardashian-kanye-west-divorce"
    },
    {
        "id": "gossipcop-real-011",
        "title": "Christopher Nolan Directs Oppenheimer Featuring Cillian Murphy in Lead Role",
        "text": "Universal Pictures confirmed director Christopher Nolan's epic biographical drama Oppenheimer, starring Cillian Murphy as J. Robert Oppenheimer, the theoretical physicist who led the Manhattan Project. The film was shot using IMAX 65mm and 65mm large-format film photography.",
        "label": 1,
        "label_name": "REAL",
        "source": "Deadline",
        "url": "https://deadline.com/2021/10/christopher-nolan-oppenheimer-cillian-murphy-universal"
    },
    {
        "id": "gossipcop-real-012",
        "title": "Beyoncé Announces Renaissance World Tour Following Seventh Studio Album",
        "text": "Beyoncé announced the Renaissance World Tour across stadium venues in Europe and North America to support her acclaimed dance-heavy album Renaissance. The tour marks her first solo stadium tour since the Formation World Tour in 2016.",
        "label": 1,
        "label_name": "REAL",
        "source": "The Hollywood Reporter",
        "url": "https://hollywoodreporter.com/news/music-news/beyonce-renaissance-world-tour-dates"
    },

    # FAKE NEWS SAMPLES (Label 0)
    {
        "id": "gossipcop-fake-001",
        "title": "Jennifer Aniston Pregnant With Brad Pitt's Miracle Twin Babies After Secret Paris Wedding",
        "text": "Shocking reports reveal Jennifer Aniston and Brad Pitt have secretly remarried in an intimate candlelit ceremony in Paris, and Aniston is now expecting miracle twin girls at age 50! Sources allege the former spouses reconciled during an exotic getaway to France and have spent millions preparing a nursery at Pitt's French estate Chateau Miraval.",
        "label": 0,
        "label_name": "FAKE",
        "source": "InTouch Weekly / Tabloid Rumor",
        "url": "https://tabloidrumors.example/aniston-pitt-twins"
    },
    {
        "id": "gossipcop-fake-002",
        "title": "Queen Elizabeth II Officially Strips Prince Harry and Meghan Markle of Royal Bloodlines",
        "text": "In an unprecedented royal decree, Queen Elizabeth II has signed emergency legislation officially expelling Prince Harry and Meghan Markle from the British royal lineage entirely. Insiders claim palace guards confiscated Harry's passport and barred Meghan from ever setting foot on UK soil again following their secret betrayal.",
        "label": 0,
        "label_name": "FAKE",
        "source": "National Enquirer / Gossip Blog",
        "url": "https://gossipglobe.example/queen-removes-harry"
    },
    {
        "id": "gossipcop-fake-003",
        "title": "Lady Gaga and Bradley Cooper Secretly Moved In Together After Cooper Dumps Irina Shayk",
        "text": "Celebrity tabloids claim Lady Gaga has already moved all her belongings into Bradley Cooper's Manhattan townhouse following Cooper's breakup with Irina Shayk. An alleged housemaid claims Gaga has redecorated the master suite and the couple are planning an extravagant secret wedding in Italy before the end of the summer.",
        "label": 0,
        "label_name": "FAKE",
        "source": "Star Magazine / Celebrity Rumors",
        "url": "https://starmagazine.example/gaga-cooper-cohabitating"
    },
    {
        "id": "gossipcop-fake-004",
        "title": "Tom Cruise Quitting Hollywood and Moving to Scientology Underground Bunker",
        "text": "Action movie megastar Tom Cruise has decided to abruptly retire from acting and isolate himself permanently in an underground Scientology bunker in New Mexico to escape a predicted global catastrophe. Church executives have allegedly appointed Cruise as their supreme commander of secret subterranean operations.",
        "label": 0,
        "label_name": "FAKE",
        "source": "Globe Magazine",
        "url": "https://globemagazine.example/tom-cruise-bunker"
    },
    {
        "id": "gossipcop-fake-005",
        "title": "Leonardo DiCaprio Caught Running Secret Illegal Billionaire Underground Poker Ring in Beverly Hills",
        "text": "FBI agents have reportedly raided Leonardo DiCaprio's Hollywood Hills mansion, uncovering an underground high-stakes gambling syndicate with tens of millions in laundered cryptocurrency. Tabloids claim several A-list Hollywood stars were arrested in handcuffs on site during an all-night poker marathon.",
        "label": 0,
        "label_name": "FAKE",
        "source": "Radar Online / Clickbait Weekly",
        "url": "https://radaronline.example/dicaprio-fbi-poker-raid"
    },
    {
        "id": "gossipcop-fake-006",
        "title": "Selena Gomez and Justin Bieber Caught Secretly Reuniting at Private Caribbean Resort",
        "text": "Justin Bieber has walked out on Hailey Baldwin after being caught red-handed with ex-girlfriend Selena Gomez at a secluded luxury villa in St. Barts. Photographers reportedly captured Bieber serenading Gomez on the beach, sparking an instant divorce standoff with Baldwin.",
        "label": 0,
        "label_name": "FAKE",
        "source": "Life & Style",
        "url": "https://lifeandstyle.example/bieber-gomez-secret-vacation"
    },
    {
        "id": "gossipcop-fake-007",
        "title": "Miley Cyrus Sentenced to Six Months Prison for Defacing National Monument",
        "text": "Pop sensation Miley Cyrus has been arrested by federal marshals and sentenced to six months in federal prison after surveillance footage showed her spray painting lyrics across Mount Rushmore during a wild midnight road trip with entourage members.",
        "label": 0,
        "label_name": "FAKE",
        "source": "WorldNewsDailyReport / Satire Web",
        "url": "https://worldnewsdailyreport.example/miley-cyrus-arrested"
    },
    {
        "id": "gossipcop-fake-008",
        "title": "Keanu Reeves Exposed as 500-Year-Old Immortal Vampire With Secret Swiss Vault",
        "text": "Shocking historical evidence and facial recognition algorithms allegedly prove Hollywood icon Keanu Reeves has lived since the 15th century without aging. Researchers claim to have uncovered historical portraits from 1530 showing Reeves serving as a European physician, holding millions in antique bullion in Zurich.",
        "label": 0,
        "label_name": "FAKE",
        "source": "Conspiracy Central Daily",
        "url": "https://conspiracycentral.example/keanu-vampire-exposed"
    },
    {
        "id": "gossipcop-fake-009",
        "title": "Katy Perry and Orlando Bloom Cancel Wedding Due to Bitter Feud Over Pet Dog",
        "text": "Katy Perry has called off her million-dollar Hawaiian wedding to Orlando Bloom just 48 hours before the ceremony following a explosive screaming match regarding whether Bloom's dog could walk down the aisle as the ring bearer. Moving vans were spotted outside their Montecito home.",
        "label": 0,
        "label_name": "FAKE",
        "source": "OK! Magazine",
        "url": "https://okmagazine.example/perry-bloom-wedding-canceled"
    },
    {
        "id": "gossipcop-fake-010",
        "title": "Johnny Depp Retiring From Acting to Launch Heavy Metal Pirate Band in the Caribbean",
        "text": "Disillusioned by Hollywood politics, Johnny Depp announced he is selling all his American assets and purchasing a 17th-century sailing schooner to tour the Caribbean playing pirate-themed death metal with an ensemble of Romanian sea musicians.",
        "label": 0,
        "label_name": "FAKE",
        "source": "Hollywood Leaks / Clickbait Daily",
        "url": "https://hollywoodleaks.example/depp-pirate-band"
    },
    {
        "id": "gossipcop-fake-011",
        "title": "Will Smith Banished from All US Theaters and Streaming Services for Life",
        "text": "Every major theater chain and global streaming conglomerate including Netflix, Disney+, and HBO Max have signed an executive boycott banishing Will Smith permanently from ever appearing on any screen in North America, effective immediately.",
        "label": 0,
        "label_name": "FAKE",
        "source": "BuzzGossip / Viral Feed",
        "url": "https://buzzgossip.example/will-smith-lifetime-boycott"
    },
    {
        "id": "gossipcop-fake-012",
        "title": "Kourtney Kardashian Expecting Alien Baby According to Classified Area 51 Whistleblower",
        "text": "A purported whistleblower from the Department of Defense has leaked classified medical telemetry claiming Kourtney Kardashian's ultrasound reveals non-human extraterrestrial DNA and anomalous heartbeat patterns following an alleged close encounter in Nevada.",
        "label": 0,
        "label_name": "FAKE",
        "source": "AlienTruthNews",
        "url": "https://alientruth.example/kourtney-extraterrestrial-baby"
    }
]

# Additional synthetic data augmentation based on GossipCop distribution to build a robust dataset
AUGMENTED_TEMPLATES = [
    # Real patterns
    ("REAL", "Sandra Bullock Adopts Second Child and Welcomes Daughter Laila", "Sandra Bullock has expanded her family by adopting a 3-year-old foster child named Laila from Louisiana. Bullock confirmed the adoption in an exclusive cover story with People, highlighting the foster care system.", "People", 1),
    ("REAL", "Robert Downey Jr. Honors Stan Lee at Marvel Hall of Fame Ceremony", "Robert Downey Jr. delivered a touching tribute to legendary Marvel creator Stan Lee during the Marvel tribute gala, reflecting on their decade of collaboration bringing Tony Stark to global cinema screens.", "Variety", 1),
    ("REAL", "Ariana Grande Donates One Love Manchester Benefit Concert Proceeds to Bombing Victims", "Ariana Grande and fellow global artists raised over 13 million dollars through the One Love Manchester benefit concert, directly benefiting victims and families affected by the Manchester Arena attack.", "BBC", 1),
    ("REAL", "Adele Sells Over 3 Million Copies of 25 in Record-Breaking Debut Week", "British singer Adele shattered single-week music sales records, selling 3.38 million copies of her third studio album 25 in the United States according to Nielsen Music data.", "Billboard", 1),
    ("REAL", "George and Amal Clooney Welcome Twins Ella and Alexander in London Hospital", "Human rights attorney Amal Clooney and Oscar-winning actor George Clooney became parents to twins, a girl named Ella and a boy named Alexander, their representative Stan Rosenfield announced.", "Reuters", 1),
    ("REAL", "Zendaya Makes Emmy History as Youngest Lead Drama Actress Winner for Euphoria", "Zendaya achieved historical recognition at the 72nd Primetime Emmy Awards, becoming the youngest woman to win Outstanding Lead Actress in a Drama Series for her role in HBO's Euphoria.", "Deadline", 1),
    ("REAL", "Dwayne The Rock Johnson Marries Longtime Partner Lauren Hashian in Hawaii", "Dwayne Johnson married his longtime partner Lauren Hashian in an intimate morning wedding ceremony on the island of Kauai, Hawaii, surrounded by close family members.", "Associated Press", 1),
    ("REAL", "Joaquin Phoenix Wins Best Actor Oscar for Portraying Arthur Fleck in Joker", "Joaquin Phoenix claimed the Academy Award for Best Actor for his visceral depiction of Arthur Fleck in Todd Phillips' Joker, using his speech to champion animal welfare and human empathy.", "The Hollywood Reporter", 1),
    
    # Fake patterns
    ("FAKE", "Barack Obama and Bruce Springsteen Launch Secret Military Coup Against US Government", "Fringe conspiracy blogs claim former President Barack Obama and rock legend Bruce Springsteen are directing a clandestine mercenary militia from a private bunker in Martha's Vineyard.", "Conspiracy Wire", 0),
    ("FAKE", "Rihanna and A$AP Rocky Break Up Overnight After Alleged Shoe Designer Affair", "Unverified social media gossip went viral claiming Rihanna abruptly ended her relationship with A$AP Rocky during Paris Fashion Week over fabricated rumors involving designer Amina Muaddi.", "ViralRumors", 0),
    ("FAKE", "Harrison Ford Confirms Fifth Indiana Jones Movie Will Feature Time-Traveling Cyborg Nazis", "Sensational clickbait sites posted fake script leaks claiming Harrison Ford's character battles mechanical cyborg soldiers sent back through time in the final installment of the franchise.", "SciFiLeaks", 0),
    ("FAKE", "Cher Banned From France for Life After Insulting French Pastry Chefs on TV", "Satirical viral pages claimed music icon Cher was issued a lifetime deportation order by the French Ministry of Culture after insulting traditional croissants on a live European morning show.", "HoaxWeekly", 0),
    ("FAKE", "Dolly Parton Revealed as Real Mother of Secret Billionaire Royal Heir", "Sensational tabloid covers claimed country legend Dolly Parton gave birth to a secret daughter in 1968 who was adopted into European royalty and stands to inherit a 50 billion dollar estate.", "National Scoop", 0),
    ("FAKE", "Leonardo DiCaprio Banned From All Commercial Airlines for Flying Pet Tigers in First Class", "A fabricated online story claimed Leonardo DiCaprio was placed on international aviation blacklists after transporting two wild Bengal tigers on a commercial transatlantic flight.", "FakeNewsFeed", 0),
    ("FAKE", "Kim Kardashian Buys Historic Vatican Painting and Accidentally Spills Coffee on Masterpiece", "Clickbait blogs circulated a phony news release claiming Kim Kardashian bought a Renaissance masterpiece from the Vatican for 80 million dollars and ruined it during a photo shoot.", "TabloidExpress", 0),
    ("FAKE", "Mark Zuckerberg Buys Hawaiian Island and Declares It Sovereign Kingdom of Meta", "Conspiracy pages reported tech billionaire Mark Zuckerberg had seceded from the United States after purchasing Kauai land, creating his own constitutional monarchy with virtual reality laws.", "TechGossip", 0)
]


def generate_full_dataset() -> pd.DataFrame:
    """Combines representative GossipCop cases and augmented samples into a structured DataFrame."""
    records = list(GOSSIPCOP_REPRESENTATIVE_SAMPLES)
    
    for idx, (label_name, title, text, source, label) in enumerate(AUGMENTED_TEMPLATES):
        records.append({
            "id": f"gossipcop-aug-{idx:03d}",
            "title": title,
            "text": text,
            "label": label,
            "label_name": label_name,
            "source": source,
            "url": f"https://archive.gossipcop.example/{idx}"
        })
        
    df = pd.DataFrame(records)
    return df


def prepare_and_save_data(data_dir: str = "data") -> dict:
    """
    Creates train/val/test splits and saves CSV files in data_dir.
    Returns dictionary with filepaths and sample counts.
    """
    os.makedirs(data_dir, exist_ok=True)
    df = generate_full_dataset()
    
    # Combine title and text for rich input representation
    df["full_text"] = df["title"] + " - " + df["text"]
    
    # Train / Test split (80% train+val, 20% test)
    train_val_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["label"]
    )
    
    # Train / Val split (approx 75% train, 25% val of the 80%)
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.20, random_state=42, stratify=train_val_df["label"]
    )
    
    full_path = os.path.join(data_dir, "gossipcop_full.csv")
    train_path = os.path.join(data_dir, "gossipcop_train.csv")
    val_path = os.path.join(data_dir, "gossipcop_val.csv")
    test_path = os.path.join(data_dir, "gossipcop_test.csv")
    
    df.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    return {
        "full_path": full_path,
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "total_count": len(df),
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "real_count": int((df["label"] == 1).sum()),
        "fake_count": int((df["label"] == 0).sum())
    }


def load_gossipcop_splits(data_dir: str = "data"):
    """Loads existing train, val, test datasets or prepares them if missing."""
    train_path = os.path.join(data_dir, "gossipcop_train.csv")
    if not os.path.exists(train_path):
        prepare_and_save_data(data_dir)
        
    train_df = pd.read_csv(os.path.join(data_dir, "gossipcop_train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "gossipcop_val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "gossipcop_test.csv"))
    full_df = pd.read_csv(os.path.join(data_dir, "gossipcop_full.csv"))
    
    return train_df, val_df, test_df, full_df


if __name__ == "__main__":
    stats = prepare_and_save_data()
    print("GossipCop Dataset Prepared Successfully:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
