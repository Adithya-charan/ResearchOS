"""model lifecycle registry, user training datasets and per-user provider selection

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-07 18:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('model_entries',
    sa.Column('model_id', sa.String(length=48), nullable=False),
    sa.Column('owner_id', sa.String(length=32), nullable=True),
    sa.Column('state', sa.String(length=16), nullable=False),
    sa.Column('stage_reached', sa.Integer(), nullable=False),
    sa.Column('evidence', sa.JSON(), nullable=False),
    sa.Column('history', sa.JSON(), nullable=False),
    sa.Column('evaluation', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('model_id')
    )
    with op.batch_alter_table('model_entries', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_model_entries_owner_id'), ['owner_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_model_entries_state'), ['state'], unique=False)

    op.create_table('training_datasets',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('owner_id', sa.String(length=32), nullable=False),
    sa.Column('name', sa.String(length=80), nullable=False),
    sa.Column('license', sa.String(length=32), nullable=False),
    sa.Column('attribution', sa.String(length=300), nullable=False),
    sa.Column('status', sa.String(length=12), nullable=False),
    sa.Column('documents', sa.Integer(), nullable=False),
    sa.Column('chars', sa.Integer(), nullable=False),
    sa.Column('manifest_sha256', sa.String(length=64), nullable=False),
    sa.Column('stats', sa.JSON(), nullable=False),
    sa.Column('error', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('training_datasets', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_training_datasets_owner_id'), ['owner_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_training_datasets_status'), ['status'], unique=False)

    op.create_table('model_selections',
    sa.Column('owner_id', sa.String(length=32), nullable=False),
    sa.Column('provider', sa.String(length=20), nullable=False),
    sa.Column('model_id', sa.String(length=48), nullable=True),
    sa.Column('mode', sa.String(length=12), nullable=False),
    sa.Column('allow_experimental', sa.Integer(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('owner_id')
    )


def downgrade() -> None:
    op.drop_table('model_selections')
    with op.batch_alter_table('training_datasets', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_training_datasets_status'))
        batch_op.drop_index(batch_op.f('ix_training_datasets_owner_id'))
    op.drop_table('training_datasets')
    with op.batch_alter_table('model_entries', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_model_entries_state'))
        batch_op.drop_index(batch_op.f('ix_model_entries_owner_id'))
    op.drop_table('model_entries')
