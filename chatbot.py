"""
Генеративный чат-бот на PyTorch
Модель учится генерировать текст посимвольно на основе диалогов
Все данные хранятся в папке data/
"""

import os
import json
import math
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
from torch.utils.data import Dataset, DataLoader


# ============================================
# КОНФИГУРАЦИЯ
# ============================================

class Config:
    """Загрузка и хранение конфигурации"""

    def __init__(self):
        # Папки и файлы
        self.data_folder = "data"
        self.dialogs_file = "dialogs.txt"
        self.model_file = "model.pth"
        self.tokenizer_file = "tokenizer.json"

        # Модель
        self.embed_dim = 128
        self.hidden_dim = 256
        self.max_len = 128

        # Обучение
        self.epochs = 150
        self.learning_rate = 0.002
        self.batch_size = 8
        self.optimizer = "adam"

        # Параметры оптимизаторов
        self.momentum = 0.9
        self.weight_decay = 0.01
        self.adam_betas = (0.9, 0.999)
        self.adam_epsilon = 1e-8

        # Планировщик LR
        self.lr_scheduler = "none"
        self.lr_step_size = 50
        self.lr_gamma = 0.5
        self.lr_patience = 10
        self.lr_factor = 0.5
        self.warmup_epochs = 10

        # Генерация
        self.temperature = 0.7
        self.max_response_len = 150

        # Логирование
        self.log_level = "DEBUG"
        self.show_test_generation = True
        self.log_interval = 10
        self.test_interval = 50

        # Интерфейс
        self.bot_name = "Бот"
        self.welcome_message = "Привет! Я готов общаться."
        self.goodbye_message = "Пока! Было приятно пообщаться."
        self.auto_load_model = False

        # Базовая директория
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

    def get_data_path(self, filename=None):
        """Получить путь к файлу в папке data"""
        data_path = os.path.join(self.base_dir, self.data_folder)
        if filename:
            return os.path.join(data_path, filename)
        return data_path

    def ensure_data_folder(self):
        """Создать папку data если её нет"""
        data_path = self.get_data_path()
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            print(f"[INFO] Создана папка: {data_path}")
        return data_path

    def load(self, filepath="config.txt"):
        """Загрузка конфига из файла"""
        full_path = os.path.join(self.base_dir, filepath)

        if not os.path.exists(full_path):
            print(f"[WARN] Конфиг не найден: {full_path}")
            print("[WARN] Используются значения по умолчанию")
            return False

        print(f"[INFO] Загрузка конфига: {full_path}")

        with open(full_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith('#'):
                    continue

                if '=' not in line:
                    continue

                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Парсим значения
                if key == "data_folder":
                    self.data_folder = value
                elif key == "dialogs_file":
                    self.dialogs_file = value
                elif key == "model_file":
                    self.model_file = value
                elif key == "tokenizer_file":
                    self.tokenizer_file = value
                elif key == "embed_dim":
                    self.embed_dim = int(value)
                elif key == "hidden_dim":
                    self.hidden_dim = int(value)
                elif key == "max_len":
                    self.max_len = int(value)
                elif key == "epochs":
                    self.epochs = int(value)
                elif key == "learning_rate":
                    self.learning_rate = float(value)
                elif key == "batch_size":
                    self.batch_size = int(value)
                elif key == "optimizer":
                    self.optimizer = value.lower()
                elif key == "momentum":
                    self.momentum = float(value)
                elif key == "weight_decay":
                    self.weight_decay = float(value)
                elif key == "adam_betas":
                    parts = value.split(',')
                    self.adam_betas = (float(parts[0].strip()), float(parts[1].strip()))
                elif key == "adam_epsilon":
                    self.adam_epsilon = float(value)
                elif key == "lr_scheduler":
                    self.lr_scheduler = value.lower()
                elif key == "lr_step_size":
                    self.lr_step_size = int(value)
                elif key == "lr_gamma":
                    self.lr_gamma = float(value)
                elif key == "lr_patience":
                    self.lr_patience = int(value)
                elif key == "lr_factor":
                    self.lr_factor = float(value)
                elif key == "warmup_epochs":
                    self.warmup_epochs = int(value)
                elif key == "temperature":
                    self.temperature = float(value)
                elif key == "max_response_len":
                    self.max_response_len = int(value)
                elif key == "log_level":
                    self.log_level = value.upper()
                elif key == "show_test_generation":
                    self.show_test_generation = value.lower() == "true"
                elif key == "log_interval":
                    self.log_interval = int(value)
                elif key == "test_interval":
                    self.test_interval = int(value)
                elif key == "bot_name":
                    self.bot_name = value
                elif key == "welcome_message":
                    self.welcome_message = value
                elif key == "goodbye_message":
                    self.goodbye_message = value
                elif key == "auto_load_model":
                    self.auto_load_model = value.lower() == "true"

        print("[INFO] Конфиг загружен успешно")
        return True

    def print_config(self):
        """Вывод текущей конфигурации"""
        print("\n[CONFIG] Текущие настройки:")
        print(f"  data_folder = {self.data_folder}")
        print(f"  dialogs_file = {self.dialogs_file}")
        print(f"  model_file = {self.model_file}")
        print(f"  tokenizer_file = {self.tokenizer_file}")
        print(f"  embed_dim = {self.embed_dim}")
        print(f"  hidden_dim = {self.hidden_dim}")
        print(f"  max_len = {self.max_len}")
        print(f"  epochs = {self.epochs}")
        print(f"  learning_rate = {self.learning_rate}")
        print(f"  batch_size = {self.batch_size}")
        print(f"  temperature = {self.temperature}")
        print(f"  optimizer = {self.optimizer}")
        print(f"  lr_scheduler = {self.lr_scheduler}")
        print(f"  log_level = {self.log_level}")
        print(f"  bot_name = {self.bot_name}")
        print()


# Глобальный конфиг
config = Config()


# ============================================
# ЛОГИРОВАНИЕ
# ============================================

LOG_LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}


def log(msg, level="INFO"):
    """Логирование с временем"""
    current_level = LOG_LEVELS.get(config.log_level, 1)
    msg_level = LOG_LEVELS.get(level, 1)

    if msg_level >= current_level:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {msg}")


def log_debug(msg):
    log(msg, "DEBUG")


def log_info(msg):
    log(msg, "INFO")


def log_warn(msg):
    log(msg, "WARN")


def log_error(msg):
    log(msg, "ERROR")


# ============================================
# ТОКЕНИЗАТОР
# ============================================

class CharTokenizer:
    """Посимвольный токенизатор с сохранением/загрузкой"""

    def __init__(self):
        self.char2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.idx2char = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.vocab_size = 3

    def fit(self, texts):
        log_debug(f"Токенизатор: обработка {len(texts)} текстов...")
        for text in texts:
            for char in text:
                if char not in self.char2idx:
                    self.char2idx[char] = self.vocab_size
                    self.idx2char[self.vocab_size] = char
                    self.vocab_size += 1
        log_info(f"Токенизатор: словарь готов, {self.vocab_size} символов")
        log_debug(f"Символы: {list(self.char2idx.keys())[:20]}...")

    def encode(self, text, max_len=None):
        encoded = [self.char2idx.get(c, 0) for c in text]
        if max_len:
            if len(encoded) < max_len:
                encoded += [0] * (max_len - len(encoded))
            else:
                encoded = encoded[:max_len]
        return encoded

    def decode(self, indices):
        chars = []
        for idx in indices:
            if idx == 2:  # <EOS>
                break
            if idx > 2:
                chars.append(self.idx2char.get(idx, ""))
        return "".join(chars)

    def save(self, filepath):
        """Сохранить токенизатор в JSON"""
        data = {
            "char2idx": self.char2idx,
            "idx2char": {str(k): v for k, v in self.idx2char.items()},
            "vocab_size": self.vocab_size
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log_info(f"Токенизатор сохранён: {os.path.basename(filepath)}")

    def load(self, filepath):
        """Загрузить токенизатор из JSON"""
        if not os.path.exists(filepath):
            log_error(f"Файл токенизатора не найден: {filepath}")
            return False

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.char2idx = data["char2idx"]
        self.idx2char = {int(k): v for k, v in data["idx2char"].items()}
        self.vocab_size = data["vocab_size"]
        log_info(f"Токенизатор загружен: {self.vocab_size} символов")
        return True


# ============================================
# ДАТАСЕТ
# ============================================

class DialogDataset(Dataset):
    """Датасет диалогов: вопрос -> ответ"""

    def __init__(self, dialogs, tokenizer, max_len):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        log_debug("Создание датасета...")

        # Собираем все тексты для токенизатора
        all_texts = []
        for q, a in dialogs:
            all_texts.append(q)
            all_texts.append(a)
        tokenizer.fit(all_texts)

        # Создаём пары (вход, выход)
        for question, answer in dialogs:
            q_enc = [1] + tokenizer.encode(question) + [2]
            a_enc = [1] + tokenizer.encode(answer) + [2]

            q_enc = q_enc[:max_len] + [0] * max(0, max_len - len(q_enc))
            a_enc = a_enc[:max_len] + [0] * max(0, max_len - len(a_enc))

            self.samples.append((q_enc[:max_len], a_enc[:max_len]))

        log_info(f"Датасет: {len(self.samples)} пар вопрос-ответ")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q, a = self.samples[idx]
        return torch.tensor(q, dtype=torch.long), torch.tensor(a, dtype=torch.long)


# ============================================
# НЕЙРОСЕТЬ
# ============================================

class Encoder(nn.Module):
    """Энкодер — кодирует входную последовательность"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)

        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = torch.tanh(self.fc(hidden))

        cell = torch.cat((cell[-2], cell[-1]), dim=1)
        cell = torch.tanh(self.fc(cell))

        return outputs, hidden.unsqueeze(0), cell.unsqueeze(0)


class Attention(nn.Module):
    """Механизм внимания"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]

        hidden = hidden.repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    """Декодер с вниманием"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim * 2, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 3 + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))

        attn_weights = self.attention(hidden.permute(1, 0, 2), encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)

        context = torch.bmm(attn_weights, encoder_outputs)

        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        output = torch.cat((output, context, embedded), dim=2)
        prediction = self.fc(output.squeeze(1))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """Полная модель Seq2Seq с вниманием"""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)

        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs

    def generate(self, src, max_len, temperature):
        """Генерация ответа"""
        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src)

            input = torch.tensor([1], device=self.device)
            generated = []

            for _ in range(max_len):
                output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)

                output = output / temperature
                probs = torch.softmax(output, dim=1)
                input = torch.multinomial(probs, 1).squeeze()

                if input.item() == 2:
                    break

                generated.append(input.item())
                input = input.unsqueeze(0) if input.dim() == 0 else input

        return generated


# ============================================
# ЧАТ-БОТ
# ============================================

class ChatBot:
    """Генеративный чат-бот"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CharTokenizer()
        self.model = None

        log_info("=" * 50)
        log_info("Инициализация чат-бота")
        log_info("=" * 50)
        log_info(f"Устройство: {self.device}")
        if self.device.type == "cuda":
            log_info(f"GPU: {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            log_info(f"Память GPU: {mem:.1f} GB")
        log_info(f"Папка данных: {config.data_folder}/")

    def _create_optimizer(self):
        """Создание оптимизатора на основе конфига"""
        opt_name = config.optimizer.lower()
        params = self.model.parameters()
        lr = config.learning_rate

        log_info(f"Оптимизатор: {opt_name.upper()}")

        if opt_name == "adam":
            optimizer = optim.Adam(
                params, lr=lr,
                betas=config.adam_betas,
                eps=config.adam_epsilon
            )
        elif opt_name == "adamw":
            optimizer = optim.AdamW(
                params, lr=lr,
                betas=config.adam_betas,
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay
            )
            log_debug(f"  weight_decay={config.weight_decay}")
        elif opt_name == "sgd":
            optimizer = optim.SGD(
                params, lr=lr,
                momentum=config.momentum
            )
            log_debug(f"  momentum={config.momentum}")
        elif opt_name == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=lr,
                momentum=config.momentum,
                eps=config.adam_epsilon
            )
            log_debug(f"  momentum={config.momentum}")
        elif opt_name == "adagrad":
            optimizer = optim.Adagrad(params, lr=lr)
        elif opt_name == "adadelta":
            optimizer = optim.Adadelta(params, lr=lr)
        elif opt_name == "nadam":
            optimizer = optim.NAdam(
                params, lr=lr,
                betas=config.adam_betas,
                eps=config.adam_epsilon
            )
        else:
            log_warn(f"Неизвестный оптимизатор '{opt_name}', используем Adam")
            optimizer = optim.Adam(params, lr=lr)

        return optimizer

    def _create_scheduler(self, optimizer):
        """Создание планировщика learning rate"""
        sched_name = config.lr_scheduler.lower()

        if sched_name == "none":
            return None

        log_info(f"LR Scheduler: {sched_name.upper()}")

        if sched_name == "step":
            scheduler = StepLR(
                optimizer,
                step_size=config.lr_step_size,
                gamma=config.lr_gamma
            )
            log_debug(f"  step_size={config.lr_step_size}, gamma={config.lr_gamma}")

        elif sched_name == "exponential":
            scheduler = ExponentialLR(
                optimizer,
                gamma=config.lr_gamma
            )
            log_debug(f"  gamma={config.lr_gamma}")

        elif sched_name == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.epochs,
                eta_min=config.learning_rate * 0.01
            )
            log_debug(f"  T_max={config.epochs}")

        elif sched_name == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.lr_factor,
                patience=config.lr_patience,
                verbose=False
            )
            log_debug(f"  patience={config.lr_patience}, factor={config.lr_factor}")

        elif sched_name == "warmup":
            warmup_epochs = config.warmup_epochs

            def warmup_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                return 1.0

            scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
            log_debug(f"  warmup_epochs={warmup_epochs}")

        elif sched_name == "warmup_cosine":
            warmup_epochs = config.warmup_epochs
            total_epochs = config.epochs

            def warmup_cosine_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
            log_debug(f"  warmup={warmup_epochs}, total={total_epochs}")

        else:
            log_warn(f"Неизвестный scheduler '{sched_name}', отключен")
            return None

        return scheduler

    def load_dialogs(self):
        """Загрузка диалогов из файла"""
        dialogs = []
        full_path = config.get_data_path(config.dialogs_file)

        log_info(f"Загрузка диалогов: {config.data_folder}/{config.dialogs_file}")

        if not os.path.exists(full_path):
            log_error(f"Файл не найден: {full_path}")
            return dialogs

        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        log_debug(f"Прочитано {len(lines)} строк из файла")

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Q:"):
                question = line[2:].strip()
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("A:"):
                    answer = lines[i + 1].strip()[2:].strip()
                    dialogs.append((question, answer))
                    i += 2
                    continue
            i += 1

        log_info(f"Загружено {len(dialogs)} диалогов")

        if dialogs:
            log_debug("Примеры диалогов:")
            for idx, (q, a) in enumerate(dialogs[:3]):
                log_debug(f"  {idx+1}. Q: {q[:30]}... -> A: {a[:30]}...")

        return dialogs

    def train(self, dialogs):
        """Обучение модели"""
        log_info("=" * 50)
        log_info("НАЧАЛО ОБУЧЕНИЯ")
        log_info("=" * 50)

        if not dialogs:
            log_error("Нет данных для обучения!")
            return

        log_info(f"Диалогов: {len(dialogs)}")
        log_info(f"Эпох: {config.epochs}")
        log_info(f"Learning rate: {config.learning_rate}")
        log_info(f"Batch size: {config.batch_size}")
        log_info(f"Embed dim: {config.embed_dim}")
        log_info(f"Hidden dim: {config.hidden_dim}")

        # Создаём датасет
        log_info("Создание датасета...")
        dataset = DialogDataset(dialogs, self.tokenizer, config.max_len)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        log_info(f"Батчей в эпохе: {len(dataloader)}")

        # Создаём модель
        log_info("Создание модели...")
        encoder = Encoder(self.tokenizer.vocab_size, config.embed_dim, config.hidden_dim)
        decoder = Decoder(self.tokenizer.vocab_size, config.embed_dim, config.hidden_dim)
        self.model = Seq2Seq(encoder, decoder, self.device).to(self.device)

        # Подсчёт параметров
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log_info(f"Параметров модели: {total_params:,}")
        log_info(f"Обучаемых параметров: {trainable_params:,}")

        # Создаём оптимизатор
        optimizer = self._create_optimizer()
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Создаём планировщик LR
        scheduler = self._create_scheduler(optimizer)

        log_info("")
        log_info("Обучение...")
        log_info("-" * 50)

        self.model.train()
        start_time = time.time()
        best_loss = float('inf')

        for epoch in range(config.epochs):
            epoch_start = time.time()
            total_loss = 0

            for src, trg in dataloader:
                src = src.to(self.device)
                trg = trg.to(self.device)

                optimizer.zero_grad()
                output = self.model(src, trg)

                output = output[:, 1:].reshape(-1, self.tokenizer.vocab_size)
                trg = trg[:, 1:].reshape(-1)

                loss = criterion(output, trg)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            epoch_time = time.time() - epoch_start

            # Логируем по интервалу или если loss улучшился
            if (epoch + 1) % config.log_interval == 0 or avg_loss < best_loss:
                status = " << NEW BEST" if avg_loss < best_loss else ""
                log_info(f"Эпоха {epoch+1:3d}/{config.epochs} | Loss: {avg_loss:.4f} | Время: {epoch_time:.1f}с{status}")

                if avg_loss < best_loss:
                    best_loss = avg_loss

            # Обновляем планировщик LR
            if scheduler is not None:
                if config.lr_scheduler == "plateau":
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()

                # Логируем текущий LR
                current_lr = optimizer.param_groups[0]['lr']
                if (epoch + 1) % config.log_interval == 0:
                    log_debug(f"Learning rate: {current_lr:.6f}")

            # Тестовая генерация
            if config.show_test_generation and (epoch + 1) % config.test_interval == 0:
                log_debug("--- Тестовая генерация ---")
                test_questions = ["привет", "как дела", "пока"]
                for q in test_questions:
                    response = self.respond(q)
                    log_debug(f"  '{q}' -> '{response}'")
                log_debug("-" * 26)
                self.model.train()

        total_time = time.time() - start_time
        log_info("-" * 50)
        log_info(f"Обучение завершено за {total_time:.1f} сек ({total_time/60:.1f} мин)")
        log_info(f"Лучший loss: {best_loss:.4f}")
        log_info("=" * 50)

    def respond(self, text, temperature=None):
        """Генерация ответа на вопрос"""
        if self.model is None:
            log_error("Модель не обучена!")
            return "Модель не обучена"

        if temperature is None:
            temperature = config.temperature

        self.model.eval()

        log_debug(f"Вход: '{text}'")

        # Кодируем вход
        encoded = [1] + self.tokenizer.encode(text) + [2]
        encoded = encoded[:config.max_len] + [0] * max(0, config.max_len - len(encoded))
        src = torch.tensor([encoded], dtype=torch.long).to(self.device)

        log_debug(f"Закодировано: {len(encoded)} токенов")

        # Генерируем
        start = time.time()
        generated = self.model.generate(src, config.max_response_len, temperature)
        gen_time = time.time() - start

        response = self.tokenizer.decode(generated)

        log_debug(f"Сгенерировано {len(generated)} токенов за {gen_time:.3f}с")
        log_debug(f"Выход: '{response}'")

        return response if response else "..."

    def chat(self):
        """Интерактивный чат"""
        print("\n" + "=" * 50)
        print("      ЧАТ-БОТ ГОТОВ К РАБОТЕ")
        print("=" * 50)
        print("Команды:")
        print("  выход/quit  - выйти")
        print("  debug       - отладка последнего ответа")
        print("  temp X      - температура (0.1-2.0)")
        print("  config      - показать конфигурацию")
        print("  reload      - перезагрузить конфиг")
        print("  files       - показать файлы данных")
        print("=" * 50)
        print(f"\n{config.bot_name}: {config.welcome_message}\n")

        temperature = config.temperature
        last_debug = ""

        while True:
            try:
                user_input = input("Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{config.bot_name}: {config.goodbye_message}")
                break

            if not user_input:
                continue

            # Команды
            if user_input.lower() in ["выход", "quit", "exit"]:
                print(f"{config.bot_name}: {config.goodbye_message}")
                break

            if user_input.lower() == "debug":
                print(f"[Отладка] {last_debug}")
                continue

            if user_input.lower() == "config":
                config.print_config()
                continue

            if user_input.lower() == "reload":
                config.load()
                temperature = config.temperature
                print("[Система] Конфиг перезагружен")
                continue

            if user_input.lower() == "files":
                self.show_data_files()
                continue

            if user_input.lower().startswith("temp "):
                try:
                    new_temp = float(user_input.split()[1])
                    if 0.1 <= new_temp <= 2.0:
                        temperature = new_temp
                        print(f"[Система] Температура: {temperature}")
                    else:
                        print("[Система] Температура должна быть от 0.1 до 2.0")
                except:
                    print("[Система] Использование: temp 0.7")
                continue

            # Генерация ответа
            start = time.time()
            response = self.respond(user_input, temperature)
            resp_time = time.time() - start

            last_debug = f"Время: {resp_time:.3f}с | Темп: {temperature} | Длина: {len(response)} симв."

            print(f"{config.bot_name}: {response}\n")

    def show_data_files(self):
        """Показать файлы в папке данных"""
        data_path = config.get_data_path()
        print(f"\n[Файлы данных] Папка: {config.data_folder}/")

        if not os.path.exists(data_path):
            print("  Папка не существует!")
            return

        files = os.listdir(data_path)
        if not files:
            print("  Папка пуста")
            return

        for f in files:
            full_path = os.path.join(data_path, f)
            size = os.path.getsize(full_path)
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.2f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  {f} ({size_str})")
        print()

    def save(self):
        """Сохранение модели и токенизатора"""
        config.ensure_data_folder()

        # Сохраняем модель
        model_path = config.get_data_path(config.model_file)
        log_info(f"Сохранение модели: {config.data_folder}/{config.model_file}")

        torch.save({
            'model_state': self.model.state_dict(),
            'embed_dim': config.embed_dim,
            'hidden_dim': config.hidden_dim,
            'max_len': config.max_len,
            'vocab_size': self.tokenizer.vocab_size
        }, model_path)

        file_size = os.path.getsize(model_path) / 1024 / 1024
        log_info(f"Модель сохранена ({file_size:.2f} MB)")

        # Сохраняем токенизатор
        tokenizer_path = config.get_data_path(config.tokenizer_file)
        self.tokenizer.save(tokenizer_path)

        log_info(f"Все данные сохранены в папку: {config.data_folder}/")

    def load(self):
        """Загрузка модели и токенизатора"""
        model_path = config.get_data_path(config.model_file)
        tokenizer_path = config.get_data_path(config.tokenizer_file)

        log_info(f"Загрузка данных из: {config.data_folder}/")

        # Проверяем файлы
        if not os.path.exists(model_path):
            log_error(f"Модель не найдена: {config.model_file}")
            return False

        if not os.path.exists(tokenizer_path):
            log_error(f"Токенизатор не найден: {config.tokenizer_file}")
            return False

        # Загружаем токенизатор
        if not self.tokenizer.load(tokenizer_path):
            return False

        # Загружаем модель
        log_info(f"Загрузка модели: {config.model_file}")
        checkpoint = torch.load(model_path, map_location=self.device)

        embed_dim = checkpoint.get('embed_dim', config.embed_dim)
        hidden_dim = checkpoint.get('hidden_dim', config.hidden_dim)
        vocab_size = checkpoint.get('vocab_size', self.tokenizer.vocab_size)

        encoder = Encoder(vocab_size, embed_dim, hidden_dim)
        decoder = Decoder(vocab_size, embed_dim, hidden_dim)
        self.model = Seq2Seq(encoder, decoder, self.device).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])

        log_info("Модель загружена успешно")
        log_info(f"Параметры: embed={embed_dim}, hidden={hidden_dim}, vocab={vocab_size}")
        return True


# ============================================
# MAIN
# ============================================

def main():
    print()
    print("=" * 50)
    print("   ГЕНЕРАТИВНЫЙ ЧАТ-БОТ (Seq2Seq + Attention)")
    print("=" * 50)
    print()

    # Загружаем конфиг
    config.load()
    config.print_config()

    # Создаём папку данных
    config.ensure_data_folder()

    # Создаём бота
    bot = ChatBot()

    # Проверяем есть ли сохранённая модель
    model_path = config.get_data_path(config.model_file)
    tokenizer_path = config.get_data_path(config.tokenizer_file)

    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        log_info(f"Найдены сохранённые данные в {config.data_folder}/")
        log_info(f"  - {config.model_file}")
        log_info(f"  - {config.tokenizer_file}")

        if config.auto_load_model:
            log_info("Автозагрузка (auto_load_model = true)")
            if bot.load():
                bot.chat()
                return
        else:
            choice = input("Загрузить сохранённую модель? (y/n): ").strip().lower()
            if choice == 'y':
                if bot.load():
                    bot.chat()
                    return

    # Загружаем диалоги и обучаем
    dialogs = bot.load_dialogs()

    if not dialogs:
        log_error(f"Не найдены диалоги в {config.data_folder}/{config.dialogs_file}")
        log_info("Создайте файл с форматом:")
        log_info("Q: вопрос")
        log_info("A: ответ")
        input("\nНажмите Enter для выхода...")
        return

    # Обучаем
    bot.train(dialogs)

    # Сохраняем
    bot.save()

    # Запускаем чат
    bot.chat()


if __name__ == "__main__":
    main()
