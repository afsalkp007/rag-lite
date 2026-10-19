# Stripe Billing & Subscription Integration Guide

## 1. Subscription Lifecycle & Webhooks
When building recurring billing systems, never rely on client-side redirects for fulfilling customer entitlements. Always listen to asynchronous Stripe webhooks to update database state.

### Key Webhook Events:
* `customer.subscription.created`: Fired immediately when a subscription is initialized. Store the `stripe_customer_id` and initial subscription tier.
* `invoice.payment_succeeded`: Confirms that payment for the billing cycle has been cleared. Grant or extend user access until `current_period_end`.
* `invoice.payment_failed`: Indicates a declined card or insufficient funds. Transition account to `past_due` state and trigger Smart Retries (Dunning email loop).
* `customer.subscription.deleted`: Fired when a customer cancels or when all automated payment retries are exhausted. Immediately revoke premium entitlements.

## 2. Proration Behavior
By default, when a customer upgrades or downgrades mid-cycle, Stripe calculates a prorated charge or credit.
* **Immediate Upgrade:** Use `proration_behavior='always_invoice'` to generate an immediate prorated charge and upgrade the tier without waiting for the cycle renewal date.
* **Deferred Downgrade:** For downgrades, set `cancel_at_period_end=True` or `proration_behavior='none'` to prevent confusing negative invoices, leaving the customer on their higher tier until the period expires.

## 3. Dispute & Chargeback Management
When a dispute occurs, Stripe freezes the contested amount plus a non-refundable $15 dispute fee.
* Submit evidence (server access logs, signed terms, delivery tracking) within the 21-day rebuttal window.
* If the chargeback is won, the disputed funds and fee are credited back to the merchant balance.
