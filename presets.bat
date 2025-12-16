@echo off
chcp 65001 >nul 2>&1
title Presets

:menu
cls
echo ========================================
echo         PRESETS CONFIG
echo ========================================
echo.
echo  1 - Fast (quick test)
echo  2 - Balanced (recommended)
echo  3 - Quality (slow, better)
echo  4 - Large (need GPU)
echo  5 - Small (weak PC)
echo  6 - Creative (high temp)
echo  7 - Precise (low temp)
echo  8 - SGD classic
echo  9 - AdamW + cosine
echo  10 - Experimental
echo.
echo  0 - Exit
echo.
echo ========================================
set /p c=Select (0-10):

if "%c%"=="0" goto end
if "%c%"=="1" goto p1
if "%c%"=="2" goto p2
if "%c%"=="3" goto p3
if "%c%"=="4" goto p4
if "%c%"=="5" goto p5
if "%c%"=="6" goto p6
if "%c%"=="7" goto p7
if "%c%"=="8" goto p8
if "%c%"=="9" goto p9
if "%c%"=="10" goto p10
goto menu

:p1
echo data_folder = data>config.txt
echo dialogs_file = dialogs.txt>>config.txt
echo model_file = model.pth>>config.txt
echo tokenizer_file = tokenizer.json>>config.txt
echo embed_dim = 64>>config.txt
echo hidden_dim = 128>>config.txt
echo max_len = 64>>config.txt
echo epochs = 50>>config.txt
echo learning_rate = 0.005>>config.txt
echo batch_size = 16>>config.txt
echo optimizer = adam>>config.txt
echo momentum = 0.9>>config.txt
echo weight_decay = 0.01>>config.txt
echo adam_betas = 0.9, 0.999>>config.txt
echo adam_epsilon = 1e-8>>config.txt
echo lr_scheduler = none>>config.txt
echo lr_step_size = 50>>config.txt
echo lr_gamma = 0.5>>config.txt
echo lr_patience = 15>>config.txt
echo lr_factor = 0.5>>config.txt
echo warmup_epochs = 5>>config.txt
echo temperature = 0.7>>config.txt
echo max_response_len = 100>>config.txt
echo log_level = INFO>>config.txt
echo show_test_generation = true>>config.txt
echo log_interval = 10>>config.txt
echo test_interval = 25>>config.txt
echo bot_name = Bot>>config.txt
echo welcome_message = Hello>>config.txt
echo goodbye_message = Bye>>config.txt
echo auto_load_model = false>>config.txt
echo Fast preset applied!
pause
goto menu

:p2
echo data_folder = data>config.txt
echo dialogs_file = dialogs.txt>>config.txt
echo model_file = model.pth>>config.txt
echo tokenizer_file = tokenizer.json>>config.txt
echo embed_dim = 128>>config.txt
echo hidden_dim = 256>>config.txt
echo max_len = 128>>config.txt
echo epochs = 150>>config.txt
echo learning_rate = 0.002>>config.txt
echo batch_size = 8>>config.txt
echo optimizer = adam>>config.txt
echo momentum = 0.9>>config.txt
echo weight_decay = 0.01>>config.txt
echo adam_betas = 0.9, 0.999>>config.txt
echo adam_epsilon = 1e-8>>config.txt
echo lr_scheduler = plateau>>config.txt
echo lr_step_size = 50>>config.txt
echo lr_gamma = 0.5>>config.txt
echo lr_patience = 15>>config.txt
echo lr_factor = 0.5>>config.txt
echo warmup_epochs = 10>>config.txt
echo temperature = 0.7>>config.txt
echo max_response_len = 150>>config.txt
echo log_level = INFO>>config.txt
echo show_test_generation = true>>config.txt
echo log_interval = 10>>config.txt
echo test_interval = 50>>config.txt
echo bot_name = Bot>>config.txt
echo welcome_message = Hello>>config.txt
echo goodbye_message = Bye>>config.txt
echo auto_load_model = false>>config.txt
echo Balanced preset applied!
pause
goto menu

:p3
echo data_folder = data>config.txt
echo dialogs_file = dialogs.txt>>config.txt
echo model_file = model.pth>>config.txt
echo tokenizer_file = tokenizer.json>>config.txt
echo embed_dim = 256>>config.txt
echo hidden_dim = 512>>config.txt
echo max_len = 256>>config.txt
echo epochs = 300>>config.txt
echo learning_rate = 0.001>>config.txt
echo batch_size = 8>>config.txt
echo optimizer = adamw>>config.txt
echo momentum = 0.9>>config.txt
echo weight_decay = 0.01>>config.txt
echo adam_betas = 0.9, 0.999>>config.txt
echo adam_epsilon = 1e-8>>config.txt
echo lr_scheduler = warmup_cosine>>config.txt
echo lr_step_size = 50>>config.txt
echo lr_gamma = 0.5>>config.txt
echo lr_patience = 20>>config.txt
echo lr_factor = 0.5>>config.txt
echo warmup_epochs = 30>>config.txt
echo temperature = 0.7>>config.txt
echo max_response_len = 200>>config.txt
echo log_level = INFO>>config.txt
echo show_test_generation = true>>config.txt
echo log_interval = 20>>config.txt
echo test_interval = 100>>config.txt
echo bot_name = Bot>>config.txt
echo welcome_message = Hello>>config.txt
echo goodbye_message = Bye>>config.txt
echo auto_load_model = false>>config.txt
echo Quality preset applied!
pause
goto menu

:p4
echo data_folder = data>config.txt
echo dialogs_file = dialogs.txt>>config.txt
echo model_file = model.pth>>config.txt
echo tokenizer_file = tokenizer.json>>config.txt
echo embed_dim = 512>>config.txt
echo hidden_dim = 1024>>config.txt
echo max_len = 256>>config.txt
echo epochs = 200>>config.txt
echo learning_rate = 0.0005>>config.txt
echo batch_size = 4>>config.txt
echo optimizer = adamw>>config.txt
echo momentum = 0.9>>config.txt
echo weight_decay = 0.02>>config.txt
echo adam_betas = 0.9, 0.98>>config.txt
echo adam_epsilon = 1e-9>>config.txt
echo lr_scheduler = warmup_cosine>>config.txt
echo lr_step_size = 50>>config.txt
echo lr_gamma = 0.5>>config.txt
echo lr_patience = 15>>config.txt
echo lr_factor = 0.5>>config.txt
echo warmup_epochs = 20>>config.txt
echo temperature = 0.7>>config.txt
echo max_response_len = 250>>config.txt
echo log_level = INFO>>config.txt
echo show_test_generation = true>>config.txt
echo log_interval = 10>>config.txt
echo test_interval = 50>>config.txt
echo bot_name = Bot>>config.txt
echo welcome_message = Hello>>config.txt
echo goodbye_message = Bye>>config.txt
echo auto_load_model = false>>config.txt
echo Large preset applied!
pause
goto menu

:p5
echo data_folder = data>config.txt
echo dialogs_file = dialogs.txt>>config.txt
echo model_file = model.pth>>config.txt
echo tokenizer_file = tokenizer.json>>config.txt
echo embed_dim = 32>>config.txt
echo hidden_dim = 64>>config.txt
echo max_len = 64>>config.txt
echo epochs = 100>>config.txt
echo learning_rate = 0.003>>config.txt
echo batch_size = 16>>config.txt
echo optimizer = adam>>config.txt
echo momentum = 0.9>>config.txt
echo weight_decay = 0.01>>config.txt
echo adam_betas = 0.9, 0.999>>config.txt
echo adam_epsilon = 1e-8>>config.txt
echo lr_scheduler = step>>config.txt
echo lr_step_size = 30>>config.txt
echo lr_gamma = 0.5>>config.txt
echo lr_patience = 10>>config.txt
echo lr_factor = 0.5>>config.txt
echo warmup_epochs = 5>>config.txt
echo temperature = 0.8>>config.txt
echo max_response_len = 80>>config.txt
echo log_level = INFO>>config.txt
echo show_test_generation = true>>config.txt
echo log_interval = 10>>config.txt
echo test_interval = 50>>config.txt
echo bot_name = Bot>>config.txt
echo welcome_message = Hello>>config.txt
echo goodbye_message = Bye>>config.txt
echo auto_load_model = false>>config.txt
echo Small preset applied!
pause
goto menu

:p6
echo data_folder = data>config.txt
echo dialogs_file = dialogs.txt>>config.txt
echo model_file = model.pth>>config.txt
echo tokenizer_file = tokenizer.json>>config.txt
echo embed_dim = 128>>config.txt
echo hidden_dim = 256>>config.txt
echo max_len = 128>>config.txt
echo epochs = 150>>config.txt
echo learning_rate = 0.002>>config.txt
echo batch_size = 8>>config.txt
echo optimizer = adam>>config.txt
echo momentum = 0.9>>config.txt
echo weight_decay = 0.01>>config.txt
echo adam_betas = 0.9, 0.999>>config.txt
echo adam_epsilon = 1e-8>>config.txt
echo lr_scheduler = plateau>>config.txt
echo lr_step_size = 50>>config.txt
echo lr_gamma = 0.5>>config.txt
echo lr_patience = 15>>config.txt
echo lr_factor = 0.5>>config.txt
echo warmup_epochs = 10>>config.txt
echo temperature = 1.2>>config.txt
echo max_response_len = 200>>config.txt
echo log_level = INFO>>config.txt
echo show_test_generation = true>>config.txt
echo log_interval = 10>>config.txt
echo test_interval = 50>>config.txt
echo bot_name = Bot>>config.txt
echo welcome_message = Hello>>config.txt
echo goodbye_message = Bye>>config.txt
echo auto_load_model = false>>config.txt
echo Creative preset applied!
pause
goto menu

:p7
echo data_folder = data>config.txt
echo dialogs_file = dialogs.txt>>config.txt
echo model_file = model.pth>>config.txt
echo tokenizer_file = tokenizer.json>>config.txt
echo embed_dim = 128>>config.txt
echo hidden_dim = 256>>config.txt
echo max_len = 128>>config.txt
echo epochs = 200>>config.txt
echo learning_rate = 0.001>>config.txt
echo batch_size = 8>>config.txt
echo optimizer = adamw>>config.txt
echo momentum = 0.9>>config.txt
echo weight_decay = 0.01>>config.txt
echo adam_betas = 0.9, 0.999>>config.txt
echo adam_epsilon = 1e-8>>config.txt
echo lr_scheduler = cosine>>config.txt
echo lr_step_size = 50>>config.txt
echo lr_gamma = 0.5>>config.txt
echo lr_patience = 15>>config.txt
echo lr_factor = 0.5>>config.txt
echo warmup_epochs = 10>>config.txt
echo temperature = 0.3>>config.txt
echo max_response_len = 150>>config.txt
echo log_level = INFO>>config.txt
echo show_test_generation = true>>config.txt
echo log_interval = 10>>config.txt
echo test_interval = 50>>config.txt
echo bot_name = Bot>>config.txt
echo welcome_message = Hello>>config.txt
echo goodbye_message = Bye>>config.txt
echo auto_load_model = false>>config.txt
echo Precise preset applied!
pause
goto menu

:p8
echo data_folder = data>config.txt
echo dialogs_file = dialogs.txt>>config.txt
echo model_file = model.pth>>config.txt
echo tokenizer_file = tokenizer.json>>config.txt
echo embed_dim = 128>>config.txt
echo hidden_dim = 256>>config.txt
echo max_len = 128>>config.txt
echo epochs = 200>>config.txt
echo learning_rate = 0.01>>config.txt
echo batch_size = 8>>config.txt
echo optimizer = sgd>>config.txt
echo momentum = 0.9>>config.txt
echo weight_decay = 0.0001>>config.txt
echo adam_betas = 0.9, 0.999>>config.txt
echo adam_epsilon = 1e-8>>config.txt
echo lr_scheduler = step>>config.txt
echo lr_step_size = 50>>config.txt
echo lr_gamma = 0.1>>config.txt
echo lr_patience = 10>>config.txt
echo lr_factor = 0.5>>config.txt
echo warmup_epochs = 10>>config.txt
echo temperature = 0.7>>config.txt
echo max_response_len = 150>>config.txt
echo log_level = DEBUG>>config.txt
echo show_test_generation = true>>config.txt
echo log_interval = 10>>config.txt
echo test_interval = 50>>config.txt
echo bot_name = Bot>>config.txt
echo welcome_message = Hello>>config.txt
echo goodbye_message = Bye>>config.txt
echo auto_load_model = false>>config.txt
echo SGD preset applied!
pause
goto menu

:p9
echo data_folder = data>config.txt
echo dialogs_file = dialogs.txt>>config.txt
echo model_file = model.pth>>config.txt
echo tokenizer_file = tokenizer.json>>config.txt
echo embed_dim = 192>>config.txt
echo hidden_dim = 384>>config.txt
echo max_len = 128>>config.txt
echo epochs = 200>>config.txt
echo learning_rate = 0.001>>config.txt
echo batch_size = 8>>config.txt
echo optimizer = adamw>>config.txt
echo momentum = 0.9>>config.txt
echo weight_decay = 0.01>>config.txt
echo adam_betas = 0.9, 0.999>>config.txt
echo adam_epsilon = 1e-8>>config.txt
echo lr_scheduler = cosine>>config.txt
echo lr_step_size = 50>>config.txt
echo lr_gamma = 0.5>>config.txt
echo lr_patience = 15>>config.txt
echo lr_factor = 0.5>>config.txt
echo warmup_epochs = 10>>config.txt
echo temperature = 0.7>>config.txt
echo max_response_len = 150>>config.txt
echo log_level = DEBUG>>config.txt
echo show_test_generation = true>>config.txt
echo log_interval = 10>>config.txt
echo test_interval = 50>>config.txt
echo bot_name = Bot>>config.txt
echo welcome_message = Hello>>config.txt
echo goodbye_message = Bye>>config.txt
echo auto_load_model = false>>config.txt
echo AdamW preset applied!
pause
goto menu

:p10
echo data_folder = data>config.txt
echo dialogs_file = dialogs.txt>>config.txt
echo model_file = model.pth>>config.txt
echo tokenizer_file = tokenizer.json>>config.txt
echo embed_dim = 160>>config.txt
echo hidden_dim = 320>>config.txt
echo max_len = 128>>config.txt
echo epochs = 250>>config.txt
echo learning_rate = 0.002>>config.txt
echo batch_size = 8>>config.txt
echo optimizer = nadam>>config.txt
echo momentum = 0.9>>config.txt
echo weight_decay = 0.005>>config.txt
echo adam_betas = 0.9, 0.98>>config.txt
echo adam_epsilon = 1e-9>>config.txt
echo lr_scheduler = warmup_cosine>>config.txt
echo lr_step_size = 50>>config.txt
echo lr_gamma = 0.5>>config.txt
echo lr_patience = 15>>config.txt
echo lr_factor = 0.5>>config.txt
echo warmup_epochs = 25>>config.txt
echo temperature = 0.8>>config.txt
echo max_response_len = 180>>config.txt
echo log_level = DEBUG>>config.txt
echo show_test_generation = true>>config.txt
echo log_interval = 10>>config.txt
echo test_interval = 50>>config.txt
echo bot_name = Bot>>config.txt
echo welcome_message = Hello>>config.txt
echo goodbye_message = Bye>>config.txt
echo auto_load_model = false>>config.txt
echo Experimental preset applied!
pause
goto menu

:end
