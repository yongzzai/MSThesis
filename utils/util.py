def get_model_args(args):
    return {'embed_dim': args.embed_dim, 'hidden_dim': args.hidden_dim,
            'num_enc_layers': args.num_enc_layers, 'num_dec_layers': args.num_dec_layers,
            'enc_dropout': args.enc_dropout, 'dec_dropout': args.dec_dropout,
            'batch_size': args.batch_size, 'epochs': args.epochs, 'lr': args.lr, 'seed': args.seed}