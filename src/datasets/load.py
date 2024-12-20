from .polyvore import PolyvoreItems, PolyvoreCompatibilityDataset, PolyvoreTripletDataset, PolyvoreFillInTheBlankDataset
from torch.utils.data import DataLoader


def load_polyvore(args):
    polyvore_items = PolyvoreItems(
        dataset_dir=args.polyvore_dir,
    )
    if args.task == 'cp':
        train_dataset = PolyvoreCompatibilityDataset(
            dataset_dir=args.polyvore_dir,
            polyvore_type=args.polyvore_type,
            split='train',
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            collate_fn=train_dataset.collate_fn,
            batch_size=args.batch_sz,
            shuffle=True,
            num_workers=args.n_workers
        )
        valid_dataset = PolyvoreCompatibilityDataset(
            dataset_dir=args.polyvore_dir,
            polyvore_type=args.polyvore_type,
            split='valid',
        )
        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            collate_fn=valid_dataset.collate_fn,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers
        )
        test_dataset = PolyvoreCompatibilityDataset(
            dataset_dir=args.polyvore_dir,
            polyvore_type=args.polyvore_type,
            split='test',
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            collate_fn=test_dataset.collate_fn,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers
        )

    elif args.task == 'cir':
        train_dataset = PolyvoreTripletDataset(
            polyvore_items=polyvore_items,
            dataset_dir=args.polyvore_dir,
            polyvore_type=args.polyvore_type,
            split='train',
            n_samples={
                'all': 8,
                'hard': 8,
            }
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            collate_fn=train_dataset.collate_fn,
            batch_size=args.batch_sz,
            shuffle=True,
            num_workers=args.n_workers
        )
        valid_dataset = PolyvoreFillInTheBlankDataset(
            polyvore_items=polyvore_items,
            dataset_dir=args.polyvore_dir,
            polyvore_type=args.polyvore_type,
            split='valid',
        )
        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            collate_fn=valid_dataset.collate_fn,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers
        )
        test_dataset = PolyvoreFillInTheBlankDataset(
            polyvore_items=polyvore_items,
            dataset_dir=args.polyvore_dir,
            polyvore_type=args.polyvore_type,
            split='test',
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            collate_fn=test_dataset.collate_fn,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers
        )
        
    return polyvore_items, train_dataloader, valid_dataloader, test_dataloader