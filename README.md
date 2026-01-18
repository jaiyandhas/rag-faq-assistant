# InvoiceKing — Invoice Management System

InvoiceKing is a full-featured, Flask-based invoice management system designed to handle real-world business workflows such as customer management, invoice generation, tax calculation, and professional PDF exports.

Originally developed as an academic project, it has been restructured using production-oriented practices including relational data modeling, modular architecture, and extensible feature design.

---

## Overview

InvoiceKing enables businesses to:

- Manage customer records  
- Create, edit, and track invoices  
- Automatically calculate taxes and totals  
- Generate professional PDF invoices  
- Track invoice payment status  
- Print invoices with optimized layouts  

**Tech stack:** Flask · SQLAlchemy · ReportLab

---

## Core Features

### Customer Management
- Create, edit, list, and delete customers  
- Optional email, phone, and address fields  
- Cascade deletion for associated invoices  
- Dropdown-based customer selection during invoice creation  

### Invoice Management
- Automatic invoice number generation (`INV-0001`, `INV-0002`, …)  
- Issue date & due date handling  
- Multiple line items per invoice  
- Configurable tax rate (0–100%)  
- Automatic subtotal, tax, and total calculations  
- Invoice status tracking (`DRAFT`, `PAID`)  

### PDF & Print
- Professional PDF generation using ReportLab  
- Clean, print-friendly layout  
- Itemized tables with totals and notes  
- Stored under `static/pdfs/`  

---

## Why This Project Matters

Invoice generation is a common but critical business workflow.  
This project focuses on building a **clean backend system** that models real-world constraints such as relational data integrity, document generation, and extensibility.

---

## Future Improvements
- User authentication & roles  
- Recurring invoices  
- Payment gateway integration  
- Reporting & analytics  
- REST API support  
- Multi-currency & internationalization  
