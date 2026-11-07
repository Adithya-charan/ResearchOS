"""claims, evidence, verifications and novelty assessments

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-07 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('claims',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('project_id', sa.String(length=32), nullable=False),
    sa.Column('run_id', sa.String(length=32), nullable=True),
    sa.Column('owner_id', sa.String(length=32), nullable=False),
    sa.Column('claim_key', sa.String(length=120), nullable=True),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('claim_type', sa.String(length=16), nullable=False),
    sa.Column('origin_agent', sa.String(length=32), nullable=False),
    sa.Column('origin_task_id', sa.String(length=64), nullable=True),
    sa.Column('source', sa.String(length=8), nullable=False),
    sa.Column('status', sa.String(length=24), nullable=False),
    sa.Column('strength', sa.JSON(), nullable=False),
    sa.Column('explanation', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('run_id', 'claim_key', name='uq_claims_run_key')
    )
    with op.batch_alter_table('claims', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_claims_owner_id'), ['owner_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_claims_project_id'), ['project_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_claims_run_id'), ['run_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_claims_status'), ['status'], unique=False)

    op.create_table('claim_evidence',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('claim_id', sa.String(length=32), nullable=False),
    sa.Column('role', sa.String(length=14), nullable=False),
    sa.Column('directness', sa.String(length=8), nullable=True),
    sa.Column('document_id', sa.String(length=64), nullable=False),
    sa.Column('passage_id', sa.String(length=80), nullable=False),
    sa.Column('page_number', sa.Integer(), nullable=True),
    sa.Column('quote', sa.Text(), nullable=False),
    sa.Column('retrieval_method', sa.String(length=20), nullable=False),
    sa.Column('retrieval_score', sa.Float(), nullable=True),
    sa.Column('coverage', sa.Float(), nullable=True),
    sa.Column('citation_valid', sa.Integer(), nullable=False),
    sa.Column('reason', sa.Text(), nullable=False),
    sa.Column('origin', sa.String(length=14), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['claim_id'], ['claims.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('claim_evidence', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_claim_evidence_claim_id'), ['claim_id'], unique=False)

    op.create_table('claim_verifications',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('claim_id', sa.String(length=32), nullable=False),
    sa.Column('round', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(length=24), nullable=False),
    sa.Column('strength', sa.JSON(), nullable=False),
    sa.Column('explanation', sa.JSON(), nullable=False),
    sa.Column('evidence', sa.JSON(), nullable=False),
    sa.Column('verifier', sa.String(length=40), nullable=False),
    sa.Column('requested_by', sa.String(length=32), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['claim_id'], ['claims.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('claim_verifications', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_claim_verifications_claim_id'), ['claim_id'], unique=False)

    op.create_table('novelty_assessments',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('project_id', sa.String(length=32), nullable=False),
    sa.Column('owner_id', sa.String(length=32), nullable=False),
    sa.Column('run_id', sa.String(length=32), nullable=True),
    sa.Column('claim_id', sa.String(length=32), nullable=True),
    sa.Column('statement', sa.Text(), nullable=False),
    sa.Column('label', sa.String(length=60), nullable=False),
    sa.Column('content', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['claim_id'], ['claims.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('novelty_assessments', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_novelty_assessments_claim_id'), ['claim_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_novelty_assessments_owner_id'), ['owner_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_novelty_assessments_project_id'), ['project_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_novelty_assessments_run_id'), ['run_id'], unique=False)


def downgrade() -> None:
    op.drop_table('novelty_assessments')
    op.drop_table('claim_verifications')
    op.drop_table('claim_evidence')
    op.drop_table('claims')
