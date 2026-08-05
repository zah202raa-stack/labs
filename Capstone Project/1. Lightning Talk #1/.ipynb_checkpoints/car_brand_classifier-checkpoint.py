"""
car_brand_classifier.py

يصنف عمود "Model" (فيه أسماء موديلات فوضوية) إلى Brand (الشركة المصنعة)
و Base Model (اسم الموديل الأساسي بدون تفاصيل الفئة/المحرك) — جاهز للاستخدام
كـ Feature Engineering قبل تدريب موديل Machine Learning (مثلاً بدل استخدام
مئات قيم Model الفريدة كـ One-Hot، تستخدم Brand أو Base Model اللي أقل تنوعاً).

الاستخدام:
    import pandas as pd
    from car_brand_classifier import add_brand_columns

    df = pd.read_csv("cars.csv")
    df = add_brand_columns(df, model_col="Model")
    # النتيجة: عمودين جدد -> df["Brand"], df["BaseModel"]

أو تصنيف نص واحد بس:
    from car_brand_classifier import classify
    classify("Rx 350 F Sport")   # -> ('Lexus', 'RX', 'Rx 350 F Sport')

ملاحظات:
- التصنيف مبني على Pattern Matching (regex) على أساس أسماء موديلات شائعة،
  مو على موديل ML مدرب — يعني سريع ومباشر بس لازم تراجع فئة
  "Other/Unclassified" و "Unclear/Data Error" يدوياً حسب بياناتك.
- إذا القيمة نص جورجي أو رموز غريبة أو ترميز Excel علمي (زي "5.30E+62")
  بترجع "Unclear/Data Error".
- زيدي أنماط جديدة بسهولة بإضافة سطر بنفس شكل: (regex, "Brand", "Base Model")
  في قائمة BRAND_BASE — الترتيب مهم، أول match ينفذ وينتهي البحث.
"""

import re, unicodedata

def strip_georgian(s):
    return re.sub(r'[\u10A0-\u10FF]+', '', s).strip()

def clean(s):
    s = strip_georgian(s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# words to drop when extracting base model (trims/engine/edition noise)
NOISE = set("""
sport limited limitedi limit edition ltd le se s sel sxl sx l lx lt ltz ls rs premium premiumi premiym
turbo turbo. hybrid hybrid. hybryd hybrid hibrid hibridi ჰიბრიდი bybrid diesel dizel benzine long
xl xle xv xe cabrio coupe wagon touring line sport. plus plug plugin plug-in advance cvt gt gti gts
gl gls glx grand special anniversary type restailing restilling rest japan germany europuli
evropuli original modeli paketi paketi. paket packet package rline r-line f-sport fsport
amg amg-paket kompressor kompresor komp cdi tdi tfsi awd 4matic quattro xdrive m-sport
avantgarde elegance luxury sportback sport-line off road trd navigation alpina
""".split())

BRAND_BASE = [
    # (regex to match at start (case-insensitive, after clean), brand, base model)
    # Toyota
    (r'^Rav ?4', 'Toyota', 'RAV4'),
    (r'^Corolla', 'Toyota', 'Corolla'),
    (r'^Camry', 'Toyota', 'Camry'),
    (r'^Yaris', 'Toyota', 'Yaris'),
    (r'^Prius C', 'Toyota', 'Prius C'),
    (r'^Prius', 'Toyota', 'Prius'),
    (r'^Aqua', 'Toyota', 'Aqua'),
    (r'^Land Cruiser Prado', 'Toyota', 'Land Cruiser Prado'),
    (r'^Land Cruiser', 'Toyota', 'Land Cruiser'),
    (r'^Highlander', 'Toyota', 'Highlander'),
    (r'^Avalon', 'Toyota', 'Avalon'),
    (r'^Sienna$|^Sienna ', 'Toyota', 'Sienna'),
    (r'^Sienta', 'Toyota', 'Sienta'),
    (r'^Tacoma', 'Toyota', 'Tacoma'),
    (r'^Tundra', 'Toyota', 'Tundra'),
    (r'^Hilux', 'Toyota', 'Hilux'),
    (r'^Fortuner', 'Toyota', 'Fortuner'),
    (r'^Auris', 'Toyota', 'Auris'),
    (r'^Vitz', 'Toyota', 'Vitz'),
    (r'^Passo', 'Toyota', 'Passo'),
    (r'^Ist', 'Toyota', 'Ist'),
    (r'^Isis', 'Toyota', 'Isis'),
    (r'^Verso$', 'Toyota', 'Verso'),
    (r'^Voxy', 'Toyota', 'Voxy'),
    (r'^Noah', 'Toyota', 'Noah'),
    (r'^Wish', 'Toyota', 'Wish'),
    (r'^Estima', 'Toyota', 'Estima'),
    (r'^Alphard', 'Toyota', 'Alphard'),
    (r'^Belta', 'Toyota', 'Belta'),
    (r'^Ractis', 'Toyota', 'Ractis'),
    (r'^Mark X', 'Toyota', 'Mark X'),
    (r'^Century$', 'Toyota', 'Century'),
    (r'^Celica', 'Toyota', 'Celica'),
    (r'^Avensis', 'Toyota', 'Avensis'),
    (r'^Venza', 'Toyota', 'Venza'),
    (r'^Sequoia', 'Toyota', 'Sequoia'),
    (r'^Chr', 'Toyota', 'C-HR'),
    (r'^Fj Cruiser', 'Toyota', 'FJ Cruiser'),
    (r'^4Runner', 'Toyota', '4Runner'),
    (r'^Harrier', 'Toyota', 'Harrier'),
    (r'^Cami$', 'Toyota', 'Cami'),
    (r'^Bb$', 'Toyota', 'bB'),
    (r'^Fun Cargo', 'Toyota', 'Fun Cargo'),
    (r'^Will ', 'Toyota', 'Will'),
    (r'^Caldina', 'Toyota', 'Caldina'),
    (r'^Corsa', 'Toyota', 'Corsa (Toyota)'),
    (r'^Altezza', 'Toyota', 'Altezza'),
    (r'^Hiace', 'Toyota', 'Hiace'),
    (r'^Grand Hiace', 'Toyota', 'Hiace'),
    (r'^Ipsum', 'Toyota', 'Ipsum'),
    (r'^Rasheen', 'Toyota', 'RAV4'),
    (r'^Vanette', 'Nissan', 'Vanette'),

    # Lexus
    (r'^Rx[ -]?\d', 'Lexus', 'RX'),
    (r'^Es \d', 'Lexus', 'ES'),
    (r'^Gs \d', 'Lexus', 'GS'),
    (r'^Is[ -]?\d|^Is-F|^Is F', 'Lexus', 'IS'),
    (r'^Ls \d', 'Lexus', 'LS'),
    (r'^Nx \d', 'Lexus', 'NX'),
    (r'^Gx \d', 'Lexus', 'GX'),
    (r'^Lx \d', 'Lexus', 'LX'),
    (r'^Ct 200', 'Lexus', 'CT200h'),
    (r'^Rc F', 'Lexus', 'RC F'),
    (r'^Hs 250', 'Lexus', 'HS250h'),

    # Honda
    (r'^Civic', 'Honda', 'Civic'),
    (r'^Accord', 'Honda', 'Accord'),
    (r'^Cr-V', 'Honda', 'CR-V'),
    (r'^Fit', 'Honda', 'Fit'),
    (r'^Hr-V', 'Honda', 'HR-V'),
    (r'^Odyssey', 'Honda', 'Odyssey'),
    (r'^Pilot', 'Honda', 'Pilot'),
    (r'^Insight', 'Honda', 'Insight'),
    (r'^Element', 'Honda', 'Element'),
    (r'^Stream', 'Honda', 'Stream'),
    (r'^Step Wagon', 'Honda', 'Step Wagon'),
    (r'^Elysion', 'Honda', 'Elysion'),
    (r'^Crosstour', 'Honda', 'Crosstour'),
    (r'^Ridgeline', 'Honda', 'Ridgeline'),
    (r'^Edix|^Edix Fr-V', 'Honda', 'Edix (FR-V)'),
    (r'^Legend', 'Honda', 'Legend'),
    (r'^Integra', 'Honda', 'Integra'),
    (r'^Passport', 'Honda', 'Passport'),
    (r'^Shuttle', 'Honda', 'Shuttle'),
    (r'^Inspire', 'Honda', 'Inspire'),
    (r'^Cr-Z', 'Honda', 'CR-Z'),
    (r'^Tlx', 'Acura', 'TLX'),
    (r'^Tsx', 'Acura', 'TSX'),
    (r'^Tl$|^Tl ', 'Acura', 'TL'),
    (r'^Mdx', 'Acura', 'MDX'),
    (r'^Rdx', 'Acura', 'RDX'),
    (r'^Ats', 'Acura', 'ATS'),

    # Nissan
    (r'^Altima', 'Nissan', 'Altima'),
    (r'^Maxima', 'Nissan', 'Maxima'),
    (r'^Sentra', 'Nissan', 'Sentra'),
    (r'^Rogue', 'Nissan', 'Rogue'),
    (r'^Murano', 'Nissan', 'Murano'),
    (r'^Pathfinder', 'Nissan', 'Pathfinder'),
    (r'^Patrol', 'Nissan', 'Patrol'),
    (r'^X-Trail', 'Nissan', 'X-Trail'),
    (r'^Juke', 'Nissan', 'Juke'),
    (r'^Note', 'Nissan', 'Note'),
    (r'^Tiida', 'Nissan', 'Tiida'),
    (r'^Micra', 'Nissan', 'Micra'),
    (r'^Qashqai', 'Nissan', 'Qashqai'),
    (r'^Navara', 'Nissan', 'Navara'),
    (r'^Terrano', 'Nissan', 'Terrano'),
    (r'^Primera', 'Nissan', 'Primera'),
    (r'^Almera', 'Nissan', 'Almera'),
    (r'^Skyline', 'Nissan', 'Skyline'),
    (r'^Cefiro', 'Nissan', 'Cefiro'),
    (r'^Presage', 'Nissan', 'Presage'),
    (r'^Serena', 'Nissan', 'Serena'),
    (r'^Elgrand', 'Nissan', 'Elgrand'),
    (r'^Wingroad', 'Nissan', 'Wingroad'),
    (r'^Bluebird', 'Nissan', 'Bluebird'),
    (r'^Latio', 'Nissan', 'Latio'),
    (r'^Lafesta', 'Nissan', 'Lafesta'),
    (r'^Kicks', 'Nissan', 'Kicks'),
    (r'^Leaf', 'Nissan', 'Leaf'),
    (r'^Versa', 'Nissan', 'Versa'),
    (r'^March', 'Nissan', 'March'),
    (r'^Moco', 'Nissan', 'Moco'),
    (r'^Caravan', 'Nissan', 'Caravan'),
    (r'^Silvia', 'Nissan', 'Silvia'),
    (r'^Fuga', 'Nissan', 'Fuga'),
    (r'^Teana', 'Nissan', 'Teana'),
    (r'^Gt-R|^Gtr', 'Nissan', 'GT-R'),
    (r'^X-Terra', 'Nissan', 'Xterra'),
    (r'^Every Landy', 'Nissan', 'Serena (Every Landy)'),

    # Infiniti
    (r'^Qx60', 'Infiniti', 'QX60'),
    (r'^Qx80', 'Infiniti', 'QX80'),
    (r'^Qx56', 'Infiniti', 'QX56'),
    (r'^Fx35', 'Infiniti', 'FX35'),
    (r'^Fx45', 'Infiniti', 'FX45'),
    (r'^G20$|^G20 ', 'Infiniti', 'G20'),
    (r'^G35', 'Infiniti', 'G35'),
    (r'^G37', 'Infiniti', 'G37'),
    (r'^G6$', 'Infiniti', 'G6'),
    (r'^M37', 'Infiniti', 'M37'),
    (r'^Ex35', 'Infiniti', 'EX35'),
    (r'^Ex37', 'Infiniti', 'EX37'),
    (r'^Jx35', 'Infiniti', 'JX35'),
    (r'^Q45', 'Infiniti', 'Q45'),
    (r'^Q50', 'Infiniti', 'Q50'),

    # Hyundai
    (r'^Elantra', 'Hyundai', 'Elantra'),
    (r'^Sonata', 'Hyundai', 'Sonata'),
    (r'^Santa Fe', 'Hyundai', 'Santa Fe'),
    (r'^Tucson', 'Hyundai', 'Tucson'),
    (r'^Accent', 'Hyundai', 'Accent'),
    (r'^Genesis', 'Hyundai', 'Genesis'),
    (r'^Grandeur', 'Hyundai', 'Grandeur'),
    (r'^Azera', 'Hyundai', 'Azera'),
    (r'^I30', 'Hyundai', 'i30'),
    (r'^I40', 'Hyundai', 'i40'),
    (r'^I20', 'Hyundai', 'i20'),
    (r'^Getz', 'Hyundai', 'Getz'),
    (r'^Veloster', 'Hyundai', 'Veloster'),
    (r'^Kona', 'Hyundai', 'Kona'),
    (r'^Ioniq', 'Hyundai', 'Ioniq'),
    (r'^Lantra', 'Hyundai', 'Lantra'),
    (r'^H1', 'Hyundai', 'H1 / Starex'),
    (r'^Ix35', 'Hyundai', 'ix35'),
    (r'^Matrix', 'Hyundai', 'Matrix'),

    # Kia
    (r'^Sportage', 'Kia', 'Sportage'),
    (r'^Sorento', 'Kia', 'Sorento'),
    (r'^Optima', 'Kia', 'Optima'),
    (r'^Rio', 'Kia', 'Rio'),
    (r'^Picanto', 'Kia', 'Picanto'),
    (r'^Cerato', 'Kia', 'Cerato'),
    (r'^Soul', 'Kia', 'Soul'),
    (r'^Ceed', 'Kia', 'Ceed'),
    (r'^Carnival', 'Kia', 'Carnival'),
    (r'^Forte', 'Kia', 'Forte'),
    (r'^Cadenza', 'Kia', 'Cadenza'),
    (r'^Niro', 'Kia', 'Niro'),
    (r'^Carens', 'Kia', 'Carens'),

    # Mercedes-Benz
    (r'^[A-Z] ?\d{2,3}', 'Mercedes-Benz', None),  # handled specially below
    (r'^Vito', 'Mercedes-Benz', 'Vito'),
    (r'^Sprinter', 'Mercedes-Benz', 'Sprinter'),
    (r'^Viano', 'Mercedes-Benz', 'Viano'),
    (r'^Citan', 'Mercedes-Benz', 'Citan'),
    (r'^Smart', 'Smart', 'Fortwo/Forfour'),
    (r'^Amg Gt', 'Mercedes-Benz', 'AMG GT'),

    # BMW
    (r'^X1', 'BMW', 'X1'),
    (r'^X3', 'BMW', 'X3'),
    (r'^X4', 'BMW', 'X4'),
    (r'^X5', 'BMW', 'X5'),
    (r'^X6', 'BMW', 'X6'),
    (r'^Z4', 'BMW', 'Z4'),
    (r'^M3', 'BMW', 'M3'),
    (r'^M4', 'BMW', 'M4'),
    (r'^M5', 'BMW', 'M5'),
    (r'^M6', 'BMW', 'M6'),
    (r'^M550', 'BMW', '5 Series'),
    (r'^I3$', 'BMW', 'i3'),
    (r'^Gti$', 'Volkswagen', 'Golf GTI'),

    # Audi
    (r'^A3', 'Audi', 'A3'),
    (r'^A4', 'Audi', 'A4'),
    (r'^A5', 'Audi', 'A5'),
    (r'^A6', 'Audi', 'A6'),
    (r'^A7', 'Audi', 'A7'),
    (r'^A8', 'Audi', 'A8'),
    (r'^Q3', 'Audi', 'Q3'),
    (r'^Q5', 'Audi', 'Q5'),
    (r'^Q7', 'Audi', 'Q7'),
    (r'^S3', 'Audi', 'S3'),
    (r'^S6', 'Audi', 'S6'),
    (r'^Rs6', 'Audi', 'RS6'),
    (r'^Tt', 'Audi', 'TT'),
    (r'^Allroad', 'Audi', 'A6 Allroad'),

    # VW
    (r'^Golf', 'Volkswagen', 'Golf'),
    (r'^Jetta', 'Volkswagen', 'Jetta'),
    (r'^Passat', 'Volkswagen', 'Passat'),
    (r'^Polo', 'Volkswagen', 'Polo'),
    (r'^Tiguan', 'Volkswagen', 'Tiguan'),
    (r'^Touareg', 'Volkswagen', 'Touareg'),
    (r'^Touran', 'Volkswagen', 'Touran'),
    (r'^Sharan', 'Volkswagen', 'Sharan'),
    (r'^Caddy', 'Volkswagen', 'Caddy'),
    (r'^New Beetle|^Beetle', 'Volkswagen', 'Beetle'),
    (r'^Bora', 'Volkswagen', 'Bora'),
    (r'^Scirocco', 'Volkswagen', 'Scirocco'),
    (r'^Eos', 'Volkswagen', 'Eos'),
    (r'^Up$|^Up ', 'Volkswagen', 'Up'),
    (r'^Vento', 'Volkswagen', 'Vento'),
    (r'^Transporter', 'Volkswagen', 'Transporter'),
    (r'^Crafter', 'Volkswagen', 'Crafter'),
    (r'^Lupo', 'Volkswagen', 'Lupo'),
    (r'^Phaeton', 'Volkswagen', 'Phaeton'),
    (r'^Cc', 'Volkswagen', 'CC'),
    (r'^Vaneo', 'Mercedes-Benz', 'Vaneo'),
    (r'^Routan', 'Volkswagen', 'Routan'),

    # Mitsubishi
    (r'^Outlander', 'Mitsubishi', 'Outlander'),
    (r'^Pajero Mini', 'Mitsubishi', 'Pajero Mini'),
    (r'^Pajero Io', 'Mitsubishi', 'Pajero iO'),
    (r'^Pajero Sport', 'Mitsubishi', 'Pajero Sport'),
    (r'^Pajero Montero', 'Mitsubishi', 'Pajero/Montero'),
    (r'^Pajero', 'Mitsubishi', 'Pajero'),
    (r'^Montero', 'Mitsubishi', 'Montero'),
    (r'^Lancer', 'Mitsubishi', 'Lancer'),
    (r'^Galant', 'Mitsubishi', 'Galant'),
    (r'^Colt', 'Mitsubishi', 'Colt'),
    (r'^Grandis', 'Mitsubishi', 'Grandis'),
    (r'^Airtrek', 'Mitsubishi', 'Airtrek'),
    (r'^Delica', 'Mitsubishi', 'Delica'),
    (r'^Rvr', 'Mitsubishi', 'RVR'),
    (r'^Mirage', 'Mitsubishi', 'Mirage'),
    (r'^L 200|^L200', 'Mitsubishi', 'L200'),
    (r'^Eclipse', 'Mitsubishi', 'Eclipse'),
    (r'^Carisma', 'Mitsubishi', 'Carisma'),
    (r'^Chariot', 'Mitsubishi', 'Chariot'),
    (r'^Coltplus', 'Mitsubishi', 'Colt Plus'),

    # Chevrolet
    (r'^Cruze', 'Chevrolet', 'Cruze'),
    (r'^Malibu', 'Chevrolet', 'Malibu'),
    (r'^Equinox', 'Chevrolet', 'Equinox'),
    (r'^Captiva', 'Chevrolet', 'Captiva'),
    (r'^Aveo', 'Chevrolet', 'Aveo'),
    (r'^Spark', 'Chevrolet', 'Spark'),
    (r'^Camaro', 'Chevrolet', 'Camaro'),
    (r'^Traverse', 'Chevrolet', 'Traverse'),
    (r'^Suburban', 'Chevrolet', 'Suburban'),
    (r'^Silverado', 'Chevrolet', 'Silverado'),
    (r'^Impala', 'Chevrolet', 'Impala'),
    (r'^Trax', 'Chevrolet', 'Trax'),
    (r'^Sonic', 'Chevrolet', 'Sonic'),
    (r'^Volt', 'Chevrolet', 'Volt'),
    (r'^Orlando', 'Chevrolet', 'Orlando'),
    (r'^Lacetti', 'Chevrolet', 'Lacetti'),
    (r'^Nubira', 'Chevrolet', 'Nubira'),
    (r'^Colorado', 'Chevrolet', 'Colorado'),
    (r'^Hhr', 'Chevrolet', 'HHR'),
    (r'^Avalanche', 'Chevrolet', 'Avalanche'),
    (r'^Corvette', 'Chevrolet', 'Corvette'),
    (r'^Gentra', 'Chevrolet', 'Gentra'),
    (r'^Matiz', 'Chevrolet', 'Matiz'),
    (r'^Kalos', 'Chevrolet', 'Kalos'),

    # Ford
    (r'^Focus', 'Ford', 'Focus'),
    (r'^Fiesta', 'Ford', 'Fiesta'),
    (r'^Fusion', 'Ford', 'Fusion'),
    (r'^Escape', 'Ford', 'Escape'),
    (r'^Explorer', 'Ford', 'Explorer'),
    (r'^Mustang', 'Ford', 'Mustang'),
    (r'^Edge', 'Ford', 'Edge'),
    (r'^Taurus', 'Ford', 'Taurus'),
    (r'^Ranger', 'Ford', 'Ranger'),
    (r'^F150|^F-150', 'Ford', 'F-150'),
    (r'^C-Max', 'Ford', 'C-Max'),
    (r'^Transit', 'Ford', 'Transit'),
    (r'^Ecosport', 'Ford', 'EcoSport'),
    (r'^Galaxy', 'Ford', 'Galaxy'),
    (r'^Ka$|^Ka ', 'Ford', 'Ka'),
    (r'^Kuga', 'Ford', 'Kuga'),
    (r'^Courier', 'Ford', 'Courier'),
    (r'^S-Max', 'Ford', 'S-Max'),
    (r'^Maverick', 'Ford', 'Maverick'),
    (r'^Escort', 'Ford', 'Escort'),
    (r'^Mondeo', 'Ford', 'Mondeo'),
    (r'^Expedition', 'Ford', 'Expedition'),

    # Jeep
    (r'^Grand Cherokee', 'Jeep', 'Grand Cherokee'),
    (r'^Cherokee', 'Jeep', 'Cherokee'),
    (r'^Wrangler', 'Jeep', 'Wrangler'),
    (r'^Compass', 'Jeep', 'Compass'),
    (r'^Patriot', 'Jeep', 'Patriot'),
    (r'^Renegade', 'Jeep', 'Renegade'),
    (r'^Liberty', 'Jeep', 'Liberty'),

    # Subaru
    (r'^Forester', 'Subaru', 'Forester'),
    (r'^Legacy', 'Subaru', 'Legacy'),
    (r'^Impreza', 'Subaru', 'Impreza'),
    (r'^Outback', 'Subaru', 'Outback'),
    (r'^Xv', 'Subaru', 'XV'),
    (r'^Crosstrek', 'Subaru', 'Crosstrek'),
    (r'^Brz', 'Subaru', 'BRZ'),
    (r'^B9 Tribeca|^Tribeca', 'Subaru', 'Tribeca'),

    # Mazda
    (r'^Mazda ?2|^Demio', 'Mazda', 'Mazda2 (Demio)'),
    (r'^Mazda ?3|^Axela', 'Mazda', 'Mazda3 (Axela)'),
    (r'^Mazda ?5|^Premacy', 'Mazda', 'Mazda5 (Premacy)'),
    (r'^Mazda ?6|^Atenza', 'Mazda', 'Mazda6 (Atenza)'),
    (r'^Cx-3', 'Mazda', 'CX-3'),
    (r'^Cx-5', 'Mazda', 'CX-5'),
    (r'^Cx-7', 'Mazda', 'CX-7'),
    (r'^Cx-9', 'Mazda', 'CX-9'),
    (r'^Mx-5', 'Mazda', 'MX-5'),
    (r'^Rx-8|^Rx8', 'Mazda', 'RX-8'),
    (r'^Millenia', 'Mazda', 'Millenia'),
    (r'^Bongo', 'Mazda', 'Bongo'),
    (r'^Eunos', 'Mazda', 'Eunos 500'),

    # Suzuki
    (r'^Grand Vitara', 'Suzuki', 'Grand Vitara'),
    (r'^Vitara', 'Suzuki', 'Vitara'),
    (r'^Swift', 'Suzuki', 'Swift'),
    (r'^Sx4', 'Suzuki', 'SX4'),
    (r'^Jimny', 'Suzuki', 'Jimny'),
    (r'^Ignis', 'Suzuki', 'Ignis'),
    (r'^Alto', 'Suzuki', 'Alto'),
    (r'^Escudo', 'Suzuki', 'Escudo'),
    (r'^Kizashi', 'Suzuki', 'Kizashi'),
    (r'^Samurai|^Sj 413', 'Suzuki', 'Samurai'),
    (r'^Aerio', 'Suzuki', 'Aerio'),
    (r'^Liana', 'Suzuki', 'Liana'),
    (r'^Cervo', 'Suzuki', 'Cervo'),

    # Land Rover
    (r'^Range Rover Evoque', 'Land Rover', 'Range Rover Evoque'),
    (r'^Range Rover Sport', 'Land Rover', 'Range Rover Sport'),
    (r'^Range Rover Velar', 'Land Rover', 'Range Rover Velar'),
    (r'^Range Rover Vogue', 'Land Rover', 'Range Rover'),
    (r'^Range Rover', 'Land Rover', 'Range Rover'),
    (r'^Land Rover Sport', 'Land Rover', 'Range Rover Sport'),
    (r'^Discovery', 'Land Rover', 'Discovery'),
    (r'^Freelander', 'Land Rover', 'Freelander'),
    (r'^Defender', 'Land Rover', 'Defender'),

    # Porsche
    (r'^Cayenne', 'Porsche', 'Cayenne'),
    (r'^Panamera', 'Porsche', 'Panamera'),
    (r'^911', 'Porsche', '911'),
    (r'^Macan', 'Porsche', 'Macan'),
    (r'^Cayman', 'Porsche', 'Cayman'),
    (r'^Boxster', 'Porsche', 'Boxster'),

    # Chrysler
    (r'^300', 'Chrysler', '300'),
    (r'^Pt Cruiser', 'Chrysler', 'PT Cruiser'),
    (r'^Pacifica', 'Chrysler', 'Pacifica'),
    (r'^Town And Country', 'Chrysler', 'Town & Country'),
    (r'^Sebring', 'Chrysler', 'Sebring'),
    (r'^Voyager', 'Chrysler', 'Voyager'),

    # Dodge / RAM
    (r'^Journey', 'Dodge', 'Journey'),
    (r'^Charger', 'Dodge', 'Charger'),
    (r'^Durango', 'Dodge', 'Durango'),
    (r'^Avenger', 'Dodge', 'Avenger'),
    (r'^Caliber', 'Dodge', 'Caliber'),
    (r'^Dart', 'Dodge', 'Dart'),
    (r'^Neon', 'Dodge', 'Neon'),
    (r'^Ram', 'Ram', 'Ram'),
    (r'^Intrepid', 'Dodge', 'Intrepid'),
    (r'^Ramcharger', 'Dodge', 'Ramcharger'),
    (r'^Challenger', 'Dodge', 'Challenger'),
    (r'^Nitro', 'Dodge', 'Nitro'),
    (r'^Crossfire', 'Chrysler', 'Crossfire'),

    # Cadillac
    (r'^Escalade', 'Cadillac', 'Escalade'),
    (r'^Cts', 'Cadillac', 'CTS'),
    (r'^Srx', 'Cadillac', 'SRX'),
    (r'^Dts', 'Cadillac', 'DTS'),
    (r'^Allante', 'Cadillac', 'Allante'),
    (r'^Catera', 'Cadillac', 'Catera'),

    # GMC
    (r'^Yukon', 'GMC', 'Yukon'),
    (r'^Acadia', 'GMC', 'Acadia'),
    (r'^Terrain', 'GMC', 'Terrain'),
    (r'^Envoy', 'GMC', 'Envoy'),
    (r'^Sierra', 'GMC', 'Sierra'),
    (r'^Vue', 'Saturn', 'Vue'),

    # Lincoln
    (r'^Navigator', 'Lincoln', 'Navigator'),
    (r'^Mkz', 'Lincoln', 'MKZ'),
    (r'^Town Car', 'Lincoln', 'Town Car'),
    (r'^Continental', 'Lincoln', 'Continental'),

    # Buick
    (r'^Enclave', 'Buick', 'Enclave'),
    (r'^Encore', 'Buick', 'Encore'),
    (r'^Regal', 'Buick', 'Regal'),

    # Jaguar
    (r'^Xf', 'Jaguar', 'XF'),
    (r'^Xj', 'Jaguar', 'XJ'),
    (r'^X-Type', 'Jaguar', 'X-Type'),
    (r'^F-Type', 'Jaguar', 'F-Type'),
    (r'^F-Pace', 'Jaguar', 'F-Pace'),
    (r'^E-Pace', 'Jaguar', 'E-Pace'),
    (r'^Xe$|^Xe ', 'Jaguar', 'XE'),
    (r'^Xk', 'Jaguar', 'XK'),
    (r'^S-Type', 'Jaguar', 'S-Type'),

    # Volvo
    (r'^Xc90', 'Volvo', 'XC90'),
    (r'^S60', 'Volvo', 'S60'),
    (r'^S70', 'Volvo', 'S70'),
    (r'^S80', 'Volvo', 'S80'),
    (r'^V50', 'Volvo', 'V50'),
    (r'^C30', 'Volvo', 'C30'),
    (r'^C70', 'Volvo', 'C70'),
    (r'^940', 'Volvo', '940'),
    (r'^960', 'Volvo', '960'),
    (r'^S40', 'Volvo', 'S40'),

    # Alfa Romeo
    (r'^147', 'Alfa Romeo', '147'),
    (r'^159', 'Alfa Romeo', '159'),
    (r'^166', 'Alfa Romeo', '166'),
    (r'^Giulietta', 'Alfa Romeo', 'Giulietta'),

    # Fiat
    (r'^500X', 'Fiat', '500X'),
    (r'^500L', 'Fiat', '500L'),
    (r'^500C', 'Fiat', '500C'),
    (r'^500', 'Fiat', '500'),
    (r'^Panda', 'Fiat', 'Panda'),
    (r'^Punto', 'Fiat', 'Punto'),
    (r'^Doblo', 'Fiat', 'Doblo'),
    (r'^Seicento', 'Fiat', 'Seicento'),
    (r'^Cinquecento', 'Fiat', 'Cinquecento'),

    # Mini
    (r'^Cooper', 'Mini', 'Cooper'),
    (r'^Countryman', 'Mini', 'Countryman'),
    (r'^Paceman', 'Mini', 'Paceman'),
    (r'^One$', 'Mini', 'One'),

    # Skoda
    (r'^Octavia', 'Skoda', 'Octavia'),
    (r'^Fabia', 'Skoda', 'Fabia'),
    (r'^Superb', 'Skoda', 'Superb'),
    (r'^Yeti', 'Skoda', 'Yeti'),

    # Renault
    (r'^Megane', 'Renault', 'Megane'),
    (r'^Clio', 'Renault', 'Clio'),
    (r'^Laguna', 'Renault', 'Laguna'),
    (r'^Scenic', 'Renault', 'Scenic'),
    (r'^Twingo', 'Renault', 'Twingo'),
    (r'^Duster', 'Renault', 'Duster'),
    (r'^Captur', 'Renault', 'Captur'),
    (r'^Kangoo', 'Renault', 'Kangoo'),
    (r'^Fluence', 'Renault', 'Fluence'),
    (r'^Sandero', 'Renault', 'Sandero'),
    (r'^Vesta', 'Lada', 'Vesta'),

    # Opel
    (r'^Astra', 'Opel', 'Astra'),
    (r'^Vectra', 'Opel', 'Vectra'),
    (r'^Zafira', 'Opel', 'Zafira'),
    (r'^Insignia', 'Opel', 'Insignia'),
    (r'^Meriva', 'Opel', 'Meriva'),
    (r'^Agila', 'Opel', 'Agila'),
    (r'^Combo', 'Opel', 'Combo'),
    (r'^Omega', 'Opel', 'Omega'),
    (r'^Frontera', 'Opel', 'Frontera'),
    (r'^Tigra', 'Opel', 'Tigra'),
    (r'^Sintra', 'Opel', 'Sintra'),
    (r'^Corsa Corsa', 'Opel', 'Corsa'),

    # Peugeot
    (r'^208', 'Peugeot', '208'),
    (r'^207', 'Peugeot', '207'),
    (r'^206', 'Peugeot', '206'),
    (r'^3008', 'Peugeot', '3008'),
    (r'^307', 'Peugeot', '307'),
    (r'^407', 'Peugeot', '407'),
    (r'^508', 'Peugeot', '508'),
    (r'^607', 'Peugeot', '607'),
    (r'^807', 'Peugeot', '807'),
    (r'^406', 'Peugeot', '406'),

    # Citroen
    (r'^C1', 'Citroen', 'C1'),
    (r'^C4', 'Citroen', 'C4'),
    (r'^C5', 'Citroen', 'C5'),
    (r'^C8', 'Citroen', 'C8'),
    (r'^Berlingo', 'Citroen', 'Berlingo'),
    (r'^Ds 4', 'Citroen', 'DS4'),

    # Seat
    (r'^Ibiza', 'Seat', 'Ibiza'),
    (r'^Leon', 'Seat', 'Leon'),

    # Isuzu
    (r'^Rodeo', 'Isuzu', 'Rodeo'),
    (r'^Vehicross', 'Isuzu', 'VehiCROSS'),
    (r'^Trooper', 'Isuzu', 'Trooper'),

    # SsangYong
    (r'^Actyon', 'SsangYong', 'Actyon'),
    (r'^Rexton', 'SsangYong', 'Rexton'),
    (r'^Korando', 'SsangYong', 'Korando'),
    (r'^Kyron', 'SsangYong', 'Kyron'),

    # UAZ/Lada/VAZ
    (r'^315', 'UAZ', '31514'),
    (r'^969', 'LuAZ', '969'),
    (r'^2101|^2103|^2105|^2106|^2107|^2109|^2111|^2121|^Niva', 'VAZ (Lada)', 'Classic/Niva'),

    # Great Wall
    (r'^H1 Gr', 'Hyundai', 'H1 / Starex'),
    (r'^H2$', 'Great Wall', 'Hover H2'),
    (r'^H3$', 'Great Wall', 'Hover H3'),
    (r'^H6$', 'Great Wall', 'Hover H6'),

    # Mercury
    (r'^Mariner', 'Mercury', 'Mariner'),
    (r'^Grand Marquis', 'Mercury', 'Grand Marquis'),
    (r'^Sable', 'Mercury', 'Sable'),

    # Maserati
    (r'^Ghibli', 'Maserati', 'Ghibli'),
    (r'^Quattroporte', 'Maserati', 'Quattroporte'),

    # Bentley
    (r'^Continental Gt', 'Bentley', 'Continental GT'),
    (r'^Mulsanne', 'Bentley', 'Mulsanne'),

    # Tesla
    (r'^Model X', 'Tesla', 'Model X'),

    # Daewoo
    (r'^Nubira', 'Daewoo', 'Nubira'),

    # Daihatsu
    (r'^Terios', 'Daihatsu', 'Terios'),
    (r'^Sirion', 'Daihatsu', 'Sirion'),
    (r'^Move', 'Daihatsu', 'Move'),
    (r'^Feroza', 'Daihatsu', 'Feroza'),
    (r'^Mira', 'Daihatsu', 'Mira'),
    (r'^Yrv', 'Daihatsu', 'YRV'),

    # Iveco
    (r'^Iveco', 'Iveco', 'Daily'),

    # Rover misc
    (r'^Scorpio', 'Ford', 'Scorpio'),

    # Additional corrections after first pass
    (r'^Frontier', 'Nissan', 'Frontier'),
    (r'^Xl7', 'Suzuki', 'XL7'),
    (r'^Quest', 'Nissan', 'Quest'),
    (r'^Tribute', 'Mazda', 'Tribute'),
    (r'^370Z', 'Nissan', '370Z'),
    (r'^Protege', 'Mazda', 'Protege'),
    (r'^Cougar', 'Mercury', 'Cougar'),
    (r'^Veracruz', 'Hyundai', 'Veracruz'),
    (r'^Galloper', 'Hyundai', 'Galloper'),
    (r'^Daimler', 'Jaguar', 'Daimler'),
    (r'^Monterey', 'Isuzu', 'Monterey'),
    (r'^Musa', 'Lancia', 'Musa'),
    (r'^Verisa', 'Toyota', 'Verisa'),
    (r'^Mpv', 'Mazda', 'MPV'),
    (r'^Stella', 'Subaru', 'Stella'),
    (r'^Minica', 'Mitsubishi', 'Minica'),
    (r'^Fred', 'Honda', 'Freed'),
    (r'^Tc$|^Tc ', 'Scion', 'tC'),
    (r'^Xd$|^Xd ', 'Scion', 'xD'),
    (r'^Virage', 'Aston Martin', 'Virage'),
    (r'^Crossroad', 'Honda', 'Crossroad'),
    (r'^R2$', 'Subaru', 'R2'),
    (r'^Gloria', 'Nissan', 'Gloria'),
    (r'^Wizard', 'Isuzu', 'Wizard'),
    (r'^Sambar', 'Subaru', 'Sambar'),
    (r'^Hustler', 'Suzuki', 'Hustler'),
    (r'^Gonow', 'Gonow', 'Unclear model'),
    (r'^T3', 'Volkswagen', 'Transporter (T3)'),
    (r'^T5', 'Volkswagen', 'Transporter (T5)'),
    (r'^Tourneo', 'Ford', 'Tourneo Connect'),
    (r'^Avella', 'Hyundai', 'Avella'),
    (r'^1500,1600', 'Mercedes-Benz', 'Classic (W110 Stufenheck)'),
]

MB_CLASS_LETTERS = {'A','B','C','E','S','G','R','V','X'}
MB_SUV_PREFIXES = ['Gla','Glc','Gle','Gls','Glk','Gl','Ml','Cls','Clk','Cl','Slk','Sl','Cla']

def classify(raw):
    c = clean(raw)
    if c == '':
        return ('Unclear/Data Error', 'Unclear', raw)

    # 1. explicit brand/base patterns (skip the generic MB catch placeholder)
    for pat, brand, base in BRAND_BASE:
        if base is None:
            continue
        if re.match(pat, c, re.IGNORECASE):
            return (brand, base, raw)

    # 2. Mercedes SUV/coupe prefixes (Gla, Glc, Gle, Gls, Glk, Gl, Ml, Cls, Clk, Cl, Slk, Sl, Cla)
    for pfx in MB_SUV_PREFIXES:
        if re.match(r'^' + pfx + r'( |\d|$)', c, re.IGNORECASE):
            return ('Mercedes-Benz', pfx.upper() + '-Class', raw)

    # 3. Mercedes letter+number classes (A160, C200, E350, S500, G350, R350, V230, X250)
    m = re.match(r'^([A-Za-z])\s?(\d{2,3})', c)
    if m and m.group(1).upper() in MB_CLASS_LETTERS:
        return ('Mercedes-Benz', m.group(1).upper() + '-Class', raw)

    # 4. explicit non-BMW pure numbers already handled above (190, Peugeot, Volvo, Alfa, Chrysler...)
    if c == '190' or c.startswith('190 '):
        return ('Mercedes-Benz', '190-Class', raw)

    # 5. BMW pure numeric series (3-digit number, first digit 1-8)
    m = re.match(r'^(\d)(\d)(\d)\b', c)
    if m:
        first = m.group(1)
        if first in list('12345678'):
            return ('BMW', f'{first}-Series', raw)

    # single/double digit or junk numeric -> unclear
    if re.match(r'^[\d.,E+\- ]+$', c):
        return ('Unclear/Data Error', 'Unclear (junk/number)', raw)

    # fallback: unknown brand, use first word as base
    first_word = c.split(' ')[0]
    return ('Other/Unclassified', first_word, raw)



def classify_series(model_series):
    """
    يطبق classify() على عمود pandas كامل.
    يرجع DataFrame فيه Brand, BaseModel, OriginalModel.
    """
    import pandas as pd
    results = [classify(m) for m in model_series]
    return pd.DataFrame(results, columns=["Brand", "BaseModel", "OriginalModel"])


def add_brand_columns(df, model_col="Model", brand_col="Brand", base_col="BaseModel"):
    """
    يضيف عمودين جدد (Brand, BaseModel) لـ DataFrame موجود عندك،
    مبنيين على عمود الموديل الفوضوي.

    مثال:
        df = add_brand_columns(df, model_col="Model")
        df[["Model", "Brand", "BaseModel"]].head()
    """
    mapped = classify_series(df[model_col])
    df = df.copy()
    df[brand_col] = mapped["Brand"].values
    df[base_col] = mapped["BaseModel"].values
    return df


if __name__ == "__main__":
    # تجربة سريعة
    samples = ["Rx 350 F Sport", "C 200 Kompressor", "520 Vanos", "Corolla Le",
               "5.30E+62", "H1 Grand Starex", "Xyz Unknown Model 123"]
    for s in samples:
        print(s, "->", classify(s))
